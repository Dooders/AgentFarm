"""
Configuration security and encryption system.

This module provides encryption, access control, audit logging, and secret
management for the hierarchical configuration system.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    from .cryptography_mock import Fernet, hashes, serialization, rsa, padding, PBKDF2HMAC

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for configuration access."""
    
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class AccessAction(Enum):
    """Types of access actions for audit logging."""
    
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    MIGRATE = "migrate"
    RELOAD = "reload"


@dataclass
class AccessControlEntry:
    """Access control entry for configuration sections."""
    
    user: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    expires_at: Optional[float] = None


@dataclass
class AuditLogEntry:
    """Audit log entry for configuration access."""
    
    timestamp: float
    user: str
    action: AccessAction
    resource: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class EncryptionKeyManager:
    """Manages encryption keys for configuration security."""
    
    def __init__(self, key_storage_path: Optional[str] = None):
        """Initialize encryption key manager.
        
        Args:
            key_storage_path: Path to store encryption keys
        """
        self.key_storage_path = Path(key_storage_path) if key_storage_path else Path("config/keys")
        self.key_storage_path.mkdir(parents=True, exist_ok=True)
        self.keys: Dict[str, bytes] = {}
        self._load_keys()
    
    def _load_keys(self) -> None:
        """Load encryption keys from storage."""
        if not self.key_storage_path.exists():
            return
        
        for key_file in self.key_storage_path.glob("*.key"):
            try:
                with open(key_file, 'rb') as f:
                    key_data = f.read()
                    key_name = key_file.stem
                    self.keys[key_name] = key_data
                    logger.debug(f"Loaded encryption key: {key_name}")
            except Exception as e:
                logger.error(f"Failed to load key {key_file}: {e}")
    
    def generate_key(self, key_name: str, key_type: str = "fernet") -> bytes:
        """Generate a new encryption key.
        
        Args:
            key_name: Name for the key
            key_type: Type of key to generate (fernet, rsa)
            
        Returns:
            Generated key
        """
        if key_type == "fernet":
            key = Fernet.generate_key()
        elif key_type == "rsa":
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            key = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise ConfigurationError(f"Unsupported key type: {key_type}")
        
        self.keys[key_name] = key
        self._save_key(key_name, key)
        
        logger.info(f"Generated {key_type} key: {key_name}")
        return key
    
    def _save_key(self, key_name: str, key: bytes) -> None:
        """Save encryption key to storage.
        
        Args:
            key_name: Name of the key
            key: Key data
        """
        key_file = self.key_storage_path / f"{key_name}.key"
        with open(key_file, 'wb') as f:
            f.write(key)
        
        # Set restrictive permissions
        os.chmod(key_file, 0o600)
    
    def get_key(self, key_name: str) -> Optional[bytes]:
        """Get encryption key by name.
        
        Args:
            key_name: Name of the key
            
        Returns:
            Key data or None if not found
        """
        return self.keys.get(key_name)
    
    def delete_key(self, key_name: str) -> bool:
        """Delete encryption key.
        
        Args:
            key_name: Name of the key
            
        Returns:
            True if deleted, False if not found
        """
        if key_name in self.keys:
            del self.keys[key_name]
            
            key_file = self.key_storage_path / f"{key_name}.key"
            if key_file.exists():
                key_file.unlink()
            
            logger.info(f"Deleted encryption key: {key_name}")
            return True
        
        return False
    
    def list_keys(self) -> List[str]:
        """List all available encryption keys.
        
        Returns:
            List of key names
        """
        return list(self.keys.keys())


class ConfigurationEncryptor:
    """Encrypts and decrypts configuration values."""
    
    def __init__(self, key_manager: EncryptionKeyManager):
        """Initialize configuration encryptor.
        
        Args:
            key_manager: Encryption key manager
        """
        self.key_manager = key_manager
        self.encryption_prefix = "ENC:"
    
    def encrypt_value(self, value: str, key_name: str) -> str:
        """Encrypt a configuration value.
        
        Args:
            value: Value to encrypt
            key_name: Name of the encryption key
            
        Returns:
            Encrypted value with prefix
        """
        key = self.key_manager.get_key(key_name)
        if not key:
            raise ConfigurationError(f"Encryption key '{key_name}' not found")
        
        try:
            fernet = Fernet(key)
            encrypted_value = fernet.encrypt(value.encode('utf-8'))
            encoded_value = base64.b64encode(encrypted_value).decode('utf-8')
            return f"{self.encryption_prefix}{encoded_value}"
        except Exception as e:
            raise ConfigurationError(f"Failed to encrypt value: {e}")
    
    def decrypt_value(self, encrypted_value: str, key_name: str) -> str:
        """Decrypt a configuration value.
        
        Args:
            encrypted_value: Encrypted value with prefix
            key_name: Name of the decryption key
            
        Returns:
            Decrypted value
        """
        if not encrypted_value.startswith(self.encryption_prefix):
            return encrypted_value  # Not encrypted
        
        key = self.key_manager.get_key(key_name)
        if not key:
            raise ConfigurationError(f"Decryption key '{key_name}' not found")
        
        try:
            encoded_value = encrypted_value[len(self.encryption_prefix):]
            encrypted_data = base64.b64decode(encoded_value)
            fernet = Fernet(key)
            decrypted_value = fernet.decrypt(encrypted_data)
            return decrypted_value.decode('utf-8')
        except Exception as e:
            raise ConfigurationError(f"Failed to decrypt value: {e}")
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted.
        
        Args:
            value: Value to check
            
        Returns:
            True if encrypted, False otherwise
        """
        return value.startswith(self.encryption_prefix)
    
    def encrypt_config_section(self, config: Dict[str, Any], key_name: str, fields: List[str]) -> Dict[str, Any]:
        """Encrypt specific fields in a configuration section.
        
        Args:
            config: Configuration dictionary
            key_name: Name of the encryption key
            fields: List of field names to encrypt
            
        Returns:
            Configuration with encrypted fields
        """
        encrypted_config = config.copy()
        
        for field in fields:
            if field in encrypted_config and isinstance(encrypted_config[field], str):
                encrypted_config[field] = self.encrypt_value(encrypted_config[field], key_name)
        
        return encrypted_config
    
    def decrypt_config_section(self, config: Dict[str, Any], key_name: str) -> Dict[str, Any]:
        """Decrypt all encrypted fields in a configuration section.
        
        Args:
            config: Configuration dictionary
            key_name: Name of the decryption key
            
        Returns:
            Configuration with decrypted fields
        """
        decrypted_config = config.copy()
        
        for key, value in decrypted_config.items():
            if isinstance(value, str) and self.is_encrypted(value):
                decrypted_config[key] = self.decrypt_value(value, key_name)
            elif isinstance(value, dict):
                decrypted_config[key] = self.decrypt_config_section(value, key_name)
        
        return decrypted_config


class AccessControlManager:
    """Manages access control for configuration sections."""
    
    def __init__(self):
        """Initialize access control manager."""
        self.access_entries: Dict[str, AccessControlEntry] = {}
        self.role_permissions: Dict[str, List[str]] = {}
        self.security_levels: Dict[str, SecurityLevel] = {}
        self._initialize_default_permissions()
    
    def _initialize_default_permissions(self) -> None:
        """Initialize default role permissions."""
        self.role_permissions = {
            "admin": ["read", "write", "delete", "encrypt", "decrypt", "migrate", "reload"],
            "developer": ["read", "write", "migrate"],
            "operator": ["read", "reload"],
            "viewer": ["read"]
        }
    
    def add_access_entry(self, resource: str, entry: AccessControlEntry) -> None:
        """Add access control entry for a resource.
        
        Args:
            resource: Resource identifier
            entry: Access control entry
        """
        self.access_entries[resource] = entry
        logger.debug(f"Added access control for {resource}: {entry.user}")
    
    def remove_access_entry(self, resource: str) -> bool:
        """Remove access control entry for a resource.
        
        Args:
            resource: Resource identifier
            
        Returns:
            True if removed, False if not found
        """
        if resource in self.access_entries:
            del self.access_entries[resource]
            logger.debug(f"Removed access control for {resource}")
            return True
        return False
    
    def check_permission(self, user: str, resource: str, action: AccessAction) -> bool:
        """Check if user has permission for action on resource.
        
        Args:
            user: User identifier
            resource: Resource identifier
            action: Action to check
            
        Returns:
            True if permission granted, False otherwise
        """
        # Check if resource has access control
        if resource not in self.access_entries:
            return True  # No access control, allow access
        
        entry = self.access_entries[resource]
        
        # Check if user matches
        if entry.user != user:
            return False
        
        # Check if entry is expired
        if entry.expires_at and time.time() > entry.expires_at:
            return False
        
        # Check role permissions
        for role in entry.roles:
            if role in self.role_permissions:
                if action.value in self.role_permissions[role]:
                    return True
        
        # Check direct permissions
        if action.value in entry.permissions:
            return True
        
        return False
    
    def set_security_level(self, resource: str, level: SecurityLevel) -> None:
        """Set security level for a resource.
        
        Args:
            resource: Resource identifier
            level: Security level
        """
        self.security_levels[resource] = level
        logger.debug(f"Set security level for {resource}: {level.value}")
    
    def get_security_level(self, resource: str) -> SecurityLevel:
        """Get security level for a resource.
        
        Args:
            resource: Resource identifier
            
        Returns:
            Security level
        """
        return self.security_levels.get(resource, SecurityLevel.PUBLIC)
    
    def add_role_permission(self, role: str, permission: str) -> None:
        """Add permission to a role.
        
        Args:
            role: Role name
            permission: Permission to add
        """
        if role not in self.role_permissions:
            self.role_permissions[role] = []
        
        if permission not in self.role_permissions[role]:
            self.role_permissions[role].append(permission)
            logger.debug(f"Added permission '{permission}' to role '{role}'")
    
    def remove_role_permission(self, role: str, permission: str) -> bool:
        """Remove permission from a role.
        
        Args:
            role: Role name
            permission: Permission to remove
            
        Returns:
            True if removed, False if not found
        """
        if role in self.role_permissions and permission in self.role_permissions[role]:
            self.role_permissions[role].remove(permission)
            logger.debug(f"Removed permission '{permission}' from role '{role}'")
            return True
        return False


class AuditLogger:
    """Logs configuration access and changes for audit purposes."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        self.log_file = log_file
        self.log_entries: List[AuditLogEntry] = []
        self.max_entries = 10000  # Maximum entries to keep in memory
        self.lock = threading.Lock()
    
    def log_access(
        self,
        user: str,
        action: AccessAction,
        resource: str,
        success: bool,
        details: Dict[str, Any] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log configuration access.
        
        Args:
            user: User identifier
            action: Action performed
            resource: Resource accessed
            success: Whether action was successful
            details: Additional details
            ip_address: IP address of user
            user_agent: User agent string
        """
        entry = AuditLogEntry(
            timestamp=time.time(),
            user=user,
            action=action,
            resource=resource,
            success=success,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        with self.lock:
            self.log_entries.append(entry)
            
            # Limit memory usage
            if len(self.log_entries) > self.max_entries:
                self.log_entries = self.log_entries[-self.max_entries:]
            
            # Write to file if configured
            if self.log_file:
                self._write_to_file(entry)
    
    def _write_to_file(self, entry: AuditLogEntry) -> None:
        """Write audit entry to file.
        
        Args:
            entry: Audit log entry
        """
        try:
            log_data = {
                "timestamp": entry.timestamp,
                "user": entry.user,
                "action": entry.action.value,
                "resource": entry.resource,
                "success": entry.success,
                "details": entry.details,
                "ip_address": entry.ip_address,
                "user_agent": entry.user_agent
            }
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_audit_logs(
        self,
        user: Optional[str] = None,
        action: Optional[AccessAction] = None,
        resource: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Get audit logs with optional filtering.
        
        Args:
            user: Filter by user
            action: Filter by action
            resource: Filter by resource
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of entries to return
            
        Returns:
            List of audit log entries
        """
        with self.lock:
            filtered_entries = self.log_entries
            
            if user:
                filtered_entries = [e for e in filtered_entries if e.user == user]
            
            if action:
                filtered_entries = [e for e in filtered_entries if e.action == action]
            
            if resource:
                filtered_entries = [e for e in filtered_entries if e.resource == resource]
            
            if start_time:
                filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]
            
            if end_time:
                filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]
            
            return filtered_entries[-limit:]
    
    def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit log statistics.
        
        Returns:
            Dictionary with audit statistics
        """
        with self.lock:
            if not self.log_entries:
                return {
                    "total_entries": 0,
                    "successful_actions": 0,
                    "failed_actions": 0,
                    "unique_users": 0,
                    "unique_resources": 0,
                    "action_counts": {}
                }
            
            successful_actions = sum(1 for e in self.log_entries if e.success)
            failed_actions = len(self.log_entries) - successful_actions
            unique_users = len(set(e.user for e in self.log_entries))
            unique_resources = len(set(e.resource for e in self.log_entries))
            
            action_counts = {}
            for entry in self.log_entries:
                action = entry.action.value
                action_counts[action] = action_counts.get(action, 0) + 1
            
            return {
                "total_entries": len(self.log_entries),
                "successful_actions": successful_actions,
                "failed_actions": failed_actions,
                "unique_users": unique_users,
                "unique_resources": unique_resources,
                "action_counts": action_counts
            }


class SecretManager:
    """Manages secrets and sensitive configuration values."""
    
    def __init__(self, key_manager: EncryptionKeyManager):
        """Initialize secret manager.
        
        Args:
            key_manager: Encryption key manager
        """
        self.key_manager = key_manager
        self.encryptor = ConfigurationEncryptor(key_manager)
        self.secrets: Dict[str, str] = {}
        self.secret_metadata: Dict[str, Dict[str, Any]] = {}
    
    def store_secret(self, name: str, value: str, key_name: str, metadata: Dict[str, Any] = None) -> None:
        """Store a secret value.
        
        Args:
            name: Secret name
            value: Secret value
            key_name: Encryption key name
            metadata: Optional metadata
        """
        encrypted_value = self.encryptor.encrypt_value(value, key_name)
        self.secrets[name] = encrypted_value
        self.secret_metadata[name] = {
            "key_name": key_name,
            "created_at": time.time(),
            "metadata": metadata or {}
        }
        
        logger.info(f"Stored secret: {name}")
    
    def retrieve_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret value.
        
        Args:
            name: Secret name
            
        Returns:
            Decrypted secret value or None if not found
        """
        if name not in self.secrets:
            return None
        
        encrypted_value = self.secrets[name]
        key_name = self.secret_metadata[name]["key_name"]
        
        try:
            return self.encryptor.decrypt_value(encrypted_value, key_name)
        except Exception as e:
            logger.error(f"Failed to retrieve secret {name}: {e}")
            return None
    
    def delete_secret(self, name: str) -> bool:
        """Delete a secret.
        
        Args:
            name: Secret name
            
        Returns:
            True if deleted, False if not found
        """
        if name in self.secrets:
            del self.secrets[name]
            del self.secret_metadata[name]
            logger.info(f"Deleted secret: {name}")
            return True
        return False
    
    def list_secrets(self) -> List[str]:
        """List all secret names.
        
        Returns:
            List of secret names
        """
        return list(self.secrets.keys())
    
    def get_secret_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a secret.
        
        Args:
            name: Secret name
            
        Returns:
            Secret metadata or None if not found
        """
        return self.secret_metadata.get(name)


class SecureConfigurationManager:
    """Secure configuration manager with encryption, access control, and audit logging."""
    
    def __init__(
        self,
        key_manager: Optional[EncryptionKeyManager] = None,
        access_control: Optional[AccessControlManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        secret_manager: Optional[SecretManager] = None
    ):
        """Initialize secure configuration manager.
        
        Args:
            key_manager: Encryption key manager
            access_control: Access control manager
            audit_logger: Audit logger
            secret_manager: Secret manager
        """
        self.key_manager = key_manager or EncryptionKeyManager()
        self.access_control = access_control or AccessControlManager()
        self.audit_logger = audit_logger or AuditLogger()
        self.secret_manager = secret_manager or SecretManager(self.key_manager)
        self.encryptor = ConfigurationEncryptor(self.key_manager)
    
    def get_config_value(
        self,
        user: str,
        resource: str,
        key: str,
        default: Any = None,
        decrypt: bool = True
    ) -> Any:
        """Get configuration value with security checks.
        
        Args:
            user: User requesting access
            resource: Resource identifier
            key: Configuration key
            default: Default value
            decrypt: Whether to decrypt encrypted values
            
        Returns:
            Configuration value
            
        Raises:
            ConfigurationError: If access denied
        """
        # Check access permission
        if not self.access_control.check_permission(user, resource, AccessAction.READ):
            self.audit_logger.log_access(
                user, AccessAction.READ, resource, False,
                details={"key": key, "reason": "access_denied"}
            )
            raise ConfigurationError(f"Access denied for user '{user}' to resource '{resource}'")
        
        # Log access
        self.audit_logger.log_access(
            user, AccessAction.READ, resource, True,
            details={"key": key}
        )
        
        # Get value (implementation depends on underlying config system)
        value = self._get_raw_value(resource, key, default)
        
        # Decrypt if requested and value is encrypted
        if decrypt and isinstance(value, str) and self.encryptor.is_encrypted(value):
            try:
                # Determine encryption key (this would need to be stored with the config)
                key_name = self._get_encryption_key_name(resource, key)
                value = self.encryptor.decrypt_value(value, key_name)
            except Exception as e:
                logger.error(f"Failed to decrypt value for {resource}.{key}: {e}")
                raise ConfigurationError(f"Failed to decrypt configuration value: {e}")
        
        return value
    
    def set_config_value(
        self,
        user: str,
        resource: str,
        key: str,
        value: Any,
        encrypt: bool = False,
        encryption_key: Optional[str] = None
    ) -> None:
        """Set configuration value with security checks.
        
        Args:
            user: User setting the value
            resource: Resource identifier
            key: Configuration key
            value: Value to set
            encrypt: Whether to encrypt the value
            encryption_key: Key to use for encryption
            
        Raises:
            ConfigurationError: If access denied or encryption fails
        """
        # Check access permission
        if not self.access_control.check_permission(user, resource, AccessAction.WRITE):
            self.audit_logger.log_access(
                user, AccessAction.WRITE, resource, False,
                details={"key": key, "reason": "access_denied"}
            )
            raise ConfigurationError(f"Access denied for user '{user}' to resource '{resource}'")
        
        # Encrypt value if requested
        if encrypt and isinstance(value, str):
            if not encryption_key:
                raise ConfigurationError("Encryption key required for encrypted values")
            
            try:
                value = self.encryptor.encrypt_value(value, encryption_key)
            except Exception as e:
                self.audit_logger.log_access(
                    user, AccessAction.ENCRYPT, resource, False,
                    details={"key": key, "error": str(e)}
                )
                raise ConfigurationError(f"Failed to encrypt value: {e}")
        
        # Set value
        try:
            self._set_raw_value(resource, key, value)
            
            # Log successful write
            self.audit_logger.log_access(
                user, AccessAction.WRITE, resource, True,
                details={"key": key, "encrypted": encrypt}
            )
        except Exception as e:
            self.audit_logger.log_access(
                user, AccessAction.WRITE, resource, False,
                details={"key": key, "error": str(e)}
            )
            raise ConfigurationError(f"Failed to set configuration value: {e}")
    
    def _get_raw_value(self, resource: str, key: str, default: Any) -> Any:
        """Get raw configuration value (to be implemented by subclasses).
        
        Args:
            resource: Resource identifier
            key: Configuration key
            default: Default value
            
        Returns:
            Raw configuration value
        """
        # This would be implemented by subclasses to integrate with actual config system
        raise NotImplementedError("Subclasses must implement _get_raw_value")
    
    def _set_raw_value(self, resource: str, key: str, value: Any) -> None:
        """Set raw configuration value (to be implemented by subclasses).
        
        Args:
            resource: Resource identifier
            key: Configuration key
            value: Value to set
        """
        # This would be implemented by subclasses to integrate with actual config system
        raise NotImplementedError("Subclasses must implement _set_raw_value")
    
    def _get_encryption_key_name(self, resource: str, key: str) -> str:
        """Get encryption key name for a configuration value.
        
        Args:
            resource: Resource identifier
            key: Configuration key
            
        Returns:
            Encryption key name
        """
        # This would be implemented to determine which key to use for decryption
        return "default"
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security system statistics.
        
        Returns:
            Dictionary with security statistics
        """
        return {
            "encryption_keys": len(self.key_manager.list_keys()),
            "access_entries": len(self.access_control.access_entries),
            "security_levels": len(self.access_control.security_levels),
            "secrets": len(self.secret_manager.list_secrets()),
            "audit_logs": self.audit_logger.get_audit_stats()
        }