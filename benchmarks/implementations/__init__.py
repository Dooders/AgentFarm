"""
Benchmark implementations package.

Avoid importing specific benchmarks here to prevent importing heavy optional
dependencies when not needed. Benchmarks are imported lazily by the CLI.
"""

# Note: Individual benchmark implementations are imported lazily by the registry
# to avoid importing heavy dependencies when not needed.
