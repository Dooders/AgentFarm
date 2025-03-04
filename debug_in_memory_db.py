"""Diagnostic script to debug in-memory database pragma settings."""

from farm.database import InMemorySimulationDatabase
from farm.core.config import SimulationConfig

def print_pragmas(db, title):
    """Print the current pragma settings for a database."""
    print(f"\n{title}")
    print("=" * len(title))
    pragmas = db.get_current_pragmas()
    for key, value in pragmas.items():
        print(f"{key}: {value}")
    return pragmas

# Test with balanced profile
print("\nTesting InMemorySimulationDatabase with 'balanced' profile")
print("--------------------------------------------------------")

config = SimulationConfig()
config.db_pragma_profile = "balanced"
db = InMemorySimulationDatabase(config=config)

# Print current settings
pragmas = print_pragmas(db, "Current pragma settings (balanced profile)")

# Print stored values
print("\nStored settings:")
print(f"pragma_profile: {db.pragma_profile}")
print(f"synchronous_mode: {db.synchronous_mode}")
print(f"journal_mode: {db.journal_mode}")

db.close()

# Test with performance profile
print("\n\nTesting InMemorySimulationDatabase with 'performance' profile")
print("------------------------------------------------------------")

config = SimulationConfig()
config.db_pragma_profile = "performance"
db = InMemorySimulationDatabase(config=config)

# Print current settings
pragmas = print_pragmas(db, "Current pragma settings (performance profile)")

# Print stored values
print("\nStored settings:")
print(f"pragma_profile: {db.pragma_profile}")
print(f"synchronous_mode: {db.synchronous_mode}")
print(f"journal_mode: {db.journal_mode}")

# Try to explicitly set synchronous mode to OFF
print("\nExplicitly setting synchronous mode to OFF")
conn = db.engine.raw_connection()
cursor = conn.cursor()
cursor.execute("PRAGMA synchronous=OFF")
cursor.close()
conn.close()

# Check if it was applied
pragmas = print_pragmas(db, "After explicit setting synchronous=OFF")

db.close()

# Test with explicit synchronous_mode override
print("\n\nTesting InMemorySimulationDatabase with explicit synchronous_mode=OFF")
print("-----------------------------------------------------------------------")

config = SimulationConfig()
config.db_pragma_profile = "balanced"
config.db_synchronous_mode = "OFF"
db = InMemorySimulationDatabase(config=config)

# Print current settings
pragmas = print_pragmas(db, "Current settings with explicit synchronous_mode=OFF")

# Print stored values
print("\nStored settings:")
print(f"pragma_profile: {db.pragma_profile}")
print(f"synchronous_mode: {db.synchronous_mode}")
print(f"journal_mode: {db.journal_mode}")

db.close()

# Test with explicit journal_mode override
print("\n\nTesting InMemorySimulationDatabase with 'balanced' profile but WAL journal_mode")
print("-----------------------------------------------------------------------------")

config = SimulationConfig()
config.db_pragma_profile = "balanced"
config.db_journal_mode = "WAL"
db = InMemorySimulationDatabase(config=config)

# Print current settings
pragmas = print_pragmas(db, "Current pragma settings (balanced profile with WAL override)")

# Print stored values
print("\nStored settings:")
print(f"pragma_profile: {config.db_pragma_profile}")
print(f"journal_mode: {db._journal_mode if hasattr(db, '_journal_mode') else 'Not stored'}")
print(f"synchronous_mode: {db._synchronous_mode if hasattr(db, '_synchronous_mode') else 'Not stored'}")

db.close() 