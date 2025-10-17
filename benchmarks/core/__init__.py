"""
Core framework for the AgentFarm benchmarking system.

This module provides the foundational components for creating, running, and analyzing
benchmarks in the AgentFarm system. It includes:

- Experiment abstractions for defining benchmark workloads
- Registry system for experiment discovery and instantiation
- Runner for orchestrating experiment execution with instrumentation
- Results handling for metrics collection and reporting
- Specification system for YAML/JSON-driven benchmark configuration

The framework follows a clean separation of concerns where experiments focus on
workload execution while the runner handles timing, instrumentation, and result
aggregation.
"""

