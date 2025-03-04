"""SQLite Pragma Performance Guide

This module documents the performance implications of different SQLite pragma settings
and provides recommendations for various workloads.

Key Pragmas
-----------

synchronous:
    - OFF: Fastest but least safe (no fsync)
    - NORMAL: Good balance (fsync at critical moments)
    - FULL: Slowest but safest (fsync at every write)
    
journal_mode:
    - DELETE: Traditional rollback journal, deleted after commit
    - TRUNCATE: Journal is truncated rather than deleted
    - PERSIST: Journal is preserved after commit
    - MEMORY: Journal stored in memory (fast but unsafe)
    - WAL: Write-Ahead Logging (good for concurrent access)
    - OFF: No journaling (dangerous)
    
cache_size:
    - Negative values are in KiB (-1000 = 1MB)
    - Larger values reduce disk I/O but increase memory usage
    
temp_store:
    - MEMORY: Store temp tables in memory (faster)
    - FILE: Store temp tables on disk (less memory usage)
    
mmap_size:
    - Size of memory-mapped I/O (0 to disable)
    - Improves read performance for large databases
    
page_size:
    - Size of database pages (512 to 65536, power of 2)
    - Larger pages reduce overhead but increase I/O size
    
busy_timeout:
    - Milliseconds to wait for locks before failing
    - Higher values improve concurrency but may cause delays

automatic_index:
    - ON: SQLite creates indexes automatically for queries
    - OFF: No automatic indexes, must create manually
    
foreign_keys:
    - ON: Enforce foreign key constraints
    - OFF: Do not enforce foreign key constraints

Recommended Profiles
-------------------

Performance Profile:
    - synchronous=OFF
    - journal_mode=MEMORY
    - cache_size=-1048576 (1GB)
    - temp_store=MEMORY
    - page_size=8192
    - automatic_index=OFF
    - busy_timeout=60000
    
Safety Profile:
    - synchronous=FULL
    - journal_mode=WAL
    - cache_size=-102400 (100MB)
    - temp_store=MEMORY
    - page_size=4096
    - automatic_index=ON
    - busy_timeout=30000
    
Balanced Profile:
    - synchronous=NORMAL
    - journal_mode=WAL
    - cache_size=-204800 (200MB)
    - temp_store=MEMORY
    - page_size=4096
    - automatic_index=ON
    - busy_timeout=30000
    
Memory-Optimized Profile:
    - synchronous=NORMAL
    - journal_mode=MEMORY
    - cache_size=-51200 (50MB)
    - temp_store=MEMORY
    - page_size=4096
    - automatic_index=OFF
    - busy_timeout=15000

Performance Implications
-----------------------

Write Performance:
    - synchronous=OFF: Up to 50x faster writes than FULL
    - journal_mode=MEMORY: Up to 10x faster than DELETE
    - cache_size: Larger cache reduces writes to disk
    - page_size: Larger pages reduce overhead for large writes
    
Read Performance:
    - mmap_size: Memory-mapped I/O improves read performance
    - cache_size: Larger cache improves read performance
    - journal_mode=WAL: Better read concurrency
    
Concurrency:
    - journal_mode=WAL: Best for concurrent reads and writes
    - busy_timeout: Higher values prevent "database is locked" errors
    
Memory Usage:
    - cache_size: Directly impacts memory usage
    - temp_store=MEMORY: Increases memory usage for complex queries
    - mmap_size: Increases memory usage for read operations
    
Data Safety:
    - synchronous=FULL: Maximum protection against corruption
    - journal_mode=WAL: Good protection with better performance
    - foreign_keys=ON: Ensures data integrity

Workload-Specific Recommendations
--------------------------------

Write-Heavy Workloads:
    - Use Performance Profile for maximum throughput
    - Consider synchronous=OFF for non-critical data
    - Use journal_mode=MEMORY for maximum performance
    - Increase cache_size to reduce disk I/O
    - Disable automatic_index to reduce overhead
    
Read-Heavy Workloads:
    - Enable memory-mapped I/O with mmap_size
    - Use journal_mode=WAL for concurrent reads
    - Increase cache_size for frequently accessed data
    - Consider synchronous=NORMAL for better read performance
    
Mixed Workloads:
    - Use Balanced Profile for good overall performance
    - journal_mode=WAL provides good read/write balance
    - synchronous=NORMAL balances safety and performance
    
Memory-Constrained Environments:
    - Use Memory-Optimized Profile
    - Reduce cache_size to minimum acceptable level
    - Monitor memory usage during operation
    
Critical Data:
    - Use Safety Profile for maximum data protection
    - Always use synchronous=FULL for critical data
    - Enable foreign_keys=ON to ensure referential integrity
    - Consider regular checkpoints with WAL mode

Runtime Adjustment
-----------------

Some pragmas can be changed at runtime without reopening the connection:
    - synchronous
    - cache_size
    - temp_store
    - foreign_keys
    - automatic_index
    - mmap_size
    
Others require reopening the connection:
    - journal_mode
    - page_size

References
---------

- SQLite Pragma Documentation: https://www.sqlite.org/pragma.html
- SQLite Performance Optimization: https://www.sqlite.org/optimization.html
- SQLite WAL Mode: https://www.sqlite.org/wal.html
"""

# Pragma profiles for different use cases
PRAGMA_PROFILES = {
    "performance": {
        "synchronous": "OFF",
        "journal_mode": "MEMORY",
        "cache_size": -1048576,  # 1GB
        "temp_store": "MEMORY",
        "page_size": 8192,
        "automatic_index": "OFF",
        "busy_timeout": 60000,
        "description": "Maximum performance for write-heavy workloads, reduced data safety"
    },
    "safety": {
        "synchronous": "FULL",
        "journal_mode": "WAL",
        "cache_size": -102400,  # 100MB
        "temp_store": "MEMORY",
        "page_size": 4096,
        "automatic_index": "ON",
        "busy_timeout": 30000,
        "description": "Maximum data safety, reduced performance"
    },
    "balanced": {
        "synchronous": "NORMAL",
        "journal_mode": "WAL",
        "cache_size": -204800,  # 200MB
        "temp_store": "MEMORY",
        "page_size": 4096,
        "automatic_index": "ON",
        "busy_timeout": 30000,
        "description": "Good balance of performance and data safety"
    },
    "memory": {
        "synchronous": "NORMAL",
        "journal_mode": "MEMORY",
        "cache_size": -51200,  # 50MB
        "temp_store": "MEMORY",
        "page_size": 4096,
        "automatic_index": "OFF",
        "busy_timeout": 15000,
        "description": "Optimized for low memory usage"
    }
}

# Pragma descriptions and performance implications
PRAGMA_INFO = {
    "synchronous": {
        "description": "Controls how aggressively SQLite uses disk barriers to prevent corruption",
        "values": {
            "OFF": {
                "performance": "Excellent",
                "safety": "Poor",
                "description": "No fsync calls, maximum performance but vulnerable to corruption"
            },
            "NORMAL": {
                "performance": "Good",
                "safety": "Moderate",
                "description": "fsync at critical moments, good balance of safety and performance"
            },
            "FULL": {
                "performance": "Poor",
                "safety": "Excellent",
                "description": "fsync at every write, maximum safety but slowest performance"
            }
        }
    },
    "journal_mode": {
        "description": "Controls how SQLite implements atomic commit and rollback",
        "values": {
            "DELETE": {
                "performance": "Poor",
                "safety": "Good",
                "description": "Traditional rollback journal, deleted after commit"
            },
            "TRUNCATE": {
                "performance": "Moderate",
                "safety": "Good",
                "description": "Journal is truncated rather than deleted, slightly faster than DELETE"
            },
            "PERSIST": {
                "performance": "Moderate",
                "safety": "Good",
                "description": "Journal is preserved after commit, faster for frequent commits"
            },
            "MEMORY": {
                "performance": "Excellent",
                "safety": "Poor",
                "description": "Journal stored in memory, very fast but unsafe for power loss"
            },
            "WAL": {
                "performance": "Good",
                "safety": "Good",
                "description": "Write-Ahead Logging, good for concurrent access"
            },
            "OFF": {
                "performance": "Excellent",
                "safety": "Very Poor",
                "description": "No journaling, dangerous but fastest possible"
            }
        }
    },
    "cache_size": {
        "description": "Number of pages to keep in memory",
        "performance_impact": "Larger values improve both read and write performance but increase memory usage"
    },
    "temp_store": {
        "description": "Storage for temporary tables",
        "values": {
            "DEFAULT": {
                "performance": "Variable",
                "memory_usage": "Variable",
                "description": "Use the compile-time default"
            },
            "FILE": {
                "performance": "Moderate",
                "memory_usage": "Low",
                "description": "Store temp tables on disk"
            },
            "MEMORY": {
                "performance": "Excellent",
                "memory_usage": "High",
                "description": "Store temp tables in memory"
            }
        }
    },
    "page_size": {
        "description": "Size of database pages in bytes",
        "performance_impact": "Larger pages reduce overhead for large writes but increase I/O size"
    },
    "mmap_size": {
        "description": "Size of memory-mapped I/O in bytes",
        "performance_impact": "Improves read performance for large databases but increases memory usage"
    },
    "busy_timeout": {
        "description": "Milliseconds to wait for locks before failing",
        "performance_impact": "Higher values improve concurrency but may cause delays"
    },
    "automatic_index": {
        "description": "Controls automatic creation of indexes",
        "values": {
            "ON": {
                "performance": "Variable",
                "description": "SQLite creates indexes automatically for queries, may improve read performance"
            },
            "OFF": {
                "performance": "Variable",
                "description": "No automatic indexes, must create manually, may improve write performance"
            }
        }
    },
    "foreign_keys": {
        "description": "Controls enforcement of foreign key constraints",
        "values": {
            "ON": {
                "performance": "Slightly Lower",
                "safety": "Better",
                "description": "Enforce foreign key constraints, ensures data integrity"
            },
            "OFF": {
                "performance": "Slightly Higher",
                "safety": "Worse",
                "description": "Do not enforce foreign key constraints, faster but less safe"
            }
        }
    }
}

def get_pragma_profile(profile_name):
    """Get pragma settings for a specific profile.
    
    Parameters
    ----------
    profile_name : str
        Name of the profile: "performance", "safety", "balanced", or "memory"
        
    Returns
    -------
    dict
        Dictionary of pragma settings for the profile
    """
    return PRAGMA_PROFILES.get(profile_name, PRAGMA_PROFILES["balanced"])

def get_pragma_info(pragma_name):
    """Get information about a specific pragma.
    
    Parameters
    ----------
    pragma_name : str
        Name of the pragma
        
    Returns
    -------
    dict
        Dictionary of information about the pragma
    """
    return PRAGMA_INFO.get(pragma_name, {"description": "No information available"})

def analyze_pragma_value(pragma_name, value):
    """Analyze the performance implications of a pragma value.
    
    Parameters
    ----------
    pragma_name : str
        Name of the pragma
    value : str or int
        Value of the pragma
        
    Returns
    -------
    dict
        Dictionary of performance analysis
    """
    pragma_info = get_pragma_info(pragma_name)
    
    if "values" in pragma_info:
        # Convert value to string and uppercase for comparison
        str_value = str(value).upper()
        return pragma_info["values"].get(str_value, {"description": "Unknown value"})
    
    return {"description": pragma_info.get("performance_impact", "No analysis available")} 