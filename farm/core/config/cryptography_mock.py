"""
Mock implementation of cryptography library for testing purposes.

This module provides mock implementations of cryptography classes when the
actual cryptography package is not available.
"""

import base64
import secrets
from typing import Union


class MockFernet:
    """Mock implementation of Fernet encryption."""
    
    def __init__(self, key: bytes):
        """Initialize mock Fernet.
        
        Args:
            key: Encryption key
        """
        self.key = key
    
    def encrypt(self, data: bytes) -> bytes:
        """Mock encrypt method.
        
        Args:
            data: Data to encrypt
            
        Returns:
            "Encrypted" data (just base64 encoded for mock)
        """
        return base64.b64encode(data)
    
    def decrypt(self, data: bytes) -> bytes:
        """Mock decrypt method.
        
        Args:
            data: Data to decrypt
            
        Returns:
            Decrypted data (just base64 decoded for mock)
        """
        return base64.b64decode(data)


class MockRSAPrivateKey:
    """Mock implementation of RSA private key."""
    
    def __init__(self):
        """Initialize mock RSA private key."""
        self.key_size = 2048
    
    def private_bytes(
        self,
        encoding,
        format,
        encryption_algorithm
    ) -> bytes:
        """Mock private_bytes method.
        
        Args:
            encoding: Encoding format
            format: Key format
            encryption_algorithm: Encryption algorithm
            
        Returns:
            Mock private key bytes
        """
        return b"-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY\n-----END PRIVATE KEY-----"


class MockSerialization:
    """Mock serialization module."""
    
    Encoding = type('Encoding', (), {'PEM': 'PEM'})()
    PrivateFormat = type('PrivateFormat', (), {'PKCS8': 'PKCS8'})()
    NoEncryption = lambda: None


class MockHazmatPrimitives:
    """Mock hazmat primitives module."""
    
    def __init__(self):
        """Initialize mock hazmat primitives."""
        self.hashes = type('hashes', (), {})()
        self.serialization = MockSerialization()
        self.asymmetric = type('asymmetric', (), {
            'rsa': type('rsa', (), {
                'generate_private_key': self._generate_private_key
            })()
        })()
        self.kdf = type('kdf', (), {
            'pbkdf2': type('pbkdf2', (), {
                'PBKDF2HMAC': self._pbkdf2hmac
            })()
        })()
    
    def _generate_private_key(self, public_exponent, key_size):
        """Mock generate_private_key method."""
        return MockRSAPrivateKey()
    
    def _pbkdf2hmac(self, algorithm, length, salt, iterations):
        """Mock PBKDF2HMAC class."""
        return type('PBKDF2HMAC', (), {
            'derive': lambda self, key_material: secrets.token_bytes(length)
        })()


# Try to import real cryptography, fall back to mock
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    # Use mock implementations
    Fernet = MockFernet
    hashes = type('hashes', (), {})()
    serialization = MockSerialization()
    rsa = type('rsa', (), {
        'generate_private_key': lambda public_exponent, key_size: MockRSAPrivateKey()
    })()
    padding = type('padding', (), {})()
    PBKDF2HMAC = lambda algorithm, length, salt, iterations: type('PBKDF2HMAC', (), {
        'derive': lambda self, key_material: secrets.token_bytes(length)
    })()