# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2025-03-24]

### Added
- Deterministic random behavior in environment and simulation components
  - Added seed support for random number generators in Environment class
  - Implemented deterministic resource regeneration logic
  - Added initialization function for NumPy and PyTorch random seeds
  - Introduced seeded random positions for agent creation
- Comprehensive Redis schemas and API documentation for AgentMemory
  - New documentation for Redis schemas covering agent actions, interactions, and states
  - Formal API specification for the AgentMemory system
  - Performance considerations and optimization strategies for Redis indexing
- Memory transition mechanism and serialization utilities
  - Hybrid age-importance based memory transition mechanism
  - New importance score calculation for memory entries
  - Utility functions for serialization/deserialization of memory entries and embedding vectors

### Changed
- Renamed system from "Agent State Memory" to "AgentMemory"
- Enhanced memory system architecture and documentation
  - Updated core concepts and implementation documents
  - Improved module structure documentation
  - Enhanced development environment setup guide
  - Added error handling strategy documentation
- Refactored agent memory system with new core components
  - Updated memory architecture with hierarchical structure
  - Improved API for system interaction
  - Enhanced codebase organization

### Removed
- Deprecated files and components from old agent memory system
- Outdated core concepts related to embeddings
- Legacy documentation and configuration files

[2025-03-24]: https://github.com/Dooders/AgentFarm/compare/main...HEAD 