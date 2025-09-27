# AgentFarm

![Project Status](https://img.shields.io/badge/status-in%20development-orange)

**AgentFarm** is an advanced simulation and computational modeling platform for exploring complex systems. Designed as a comprehensive workbench, it enables users to run, analyze, and compare simulations with ease.

This repository is being developed to support research in the [Dooders](https://github.com/Dooders) project, focusing on complex adaptive systems and agent-based modeling approaches.

> **Note**: This project is currently in active development. APIs and features may change between releases. See the [Contributing Guidelines](CONTRIBUTING.md) for information on getting involved.

## Key Features

### Agent-Based Modeling & Analysis
- Run complex simulations with interacting, adaptive agents
- Study emergent behaviors and system dynamics
- Track agent interactions and environmental influences
- Analyze trends and patterns over time

### Customization & Flexibility
- Define custom parameters, rules, and environments
- Create specialized agent behaviors and properties
- Configure simulation parameters and conditions
- Design custom experiments and scenarios

### AI & Machine Learning
- Reinforcement learning for agent adaptation
- Automated data analysis and insight generation
- Pattern recognition and behavior prediction
- Evolutionary algorithms and genetic modeling

### Data & Visualization
- Comprehensive data collection and metrics
- Interactive results dashboard
- Real-time visualization tools
- Automated report generation

### Research Tools
- Parameter sweep experiments
- Comparative analysis framework
- Experiment replication tools
- Detailed logging and tracking

### Data System
- **Comprehensive Data Architecture**: Layered system with database, repositories, analyzers, and services
- **Advanced Analytics**: Action statistics, behavioral clustering, causal analysis, and pattern recognition
- **Flexible Data Access**: Repository pattern for efficient data retrieval and querying
- **High-Level Services**: Coordinated analysis operations with built-in error handling
- **Multi-Simulation Support**: Experiment database for comparing multiple simulation runs

### Accessibility & User Experience
- **Comprehensive Accessibility Support**: WCAG 2.1 AA compliant interface
- **Keyboard Navigation**: Full keyboard support with arrow keys, tab navigation, and shortcuts
- **Screen Reader Compatibility**: Proper ARIA labels, roles, and live regions
- **High Contrast Mode**: Enhanced visibility for users with visual impairments
- **Focus Management**: Proper focus trapping and restoration for modals and dynamic content
- **Skip Navigation Links**: Quick access to main content, validation errors, and comparison tools

### Additional Tools
- **Interactive Notebooks**: Jupyter notebooks for data exploration and analysis
- **Web Dashboard**: Browser-based interface for monitoring and visualization
- **Benchmarking Suite**: Performance testing and optimization tools
- **Research Tools**: Advanced analysis modules for academic research
- **Genome Embeddings**: Machine learning tools for agent evolution analysis

### IPC Service Layer
- **Comprehensive IPC Communication**: Full TypeScript implementation with type safety
- **Configuration Management**: Load, save, export, and import simulation configurations
- **File System Operations**: Complete file and directory operations with backup support
- **Template Management**: Save, load, and manage configuration templates
- **Settings Persistence**: Application settings and UI state management
- **Error Handling**: Robust error handling with retry logic and graceful fallbacks
- **Performance Monitoring**: Built-in metrics tracking and optimization
- **Cross-Platform Support**: Works in both Electron and browser environments

## Quick Start

### Prerequisites
- Python 3.8 or higher (3.9+ recommended for best performance)
- pip (Python package installer)
- Git
- Redis (optional, for enhanced memory management)

### Installation

```bash
# Clone the repository
git clone https://github.com/Dooders/AgentFarm.git
cd AgentFarm

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Install Node.js dependencies for the Live Simulation Config Explorer
npm install

# Optional: Install Redis for enhanced memory management
# Note: Redis is used for agent memory storage and can improve performance
# On Ubuntu/Debian: sudo apt-get install redis-server
# On macOS: brew install redis
# On Windows: Download from https://redis.io/download
# Then start Redis: redis-server
```

### Running Your First Simulation

AgentFarm provides multiple ways to run simulations:

**Command Line (Simple)**
```bash
python run_simulation.py --config config.yaml --steps 1000
```

**Command Line Interface (Advanced)**
```bash
# Run simulation with various options
python farm/core/cli.py --mode simulate --config config.yaml --steps 1000

# Run experiments with parameter variations
python farm/core/cli.py --mode experiment --config config.yaml --experiment-name test --iterations 3

# Visualize existing simulation results
python farm/core/cli.py --mode visualize --db-path simulations/simulation.db

# Generate analysis reports
python farm/core/cli.py --mode analyze --db-path simulations/simulation.db
```

**GUI Interface**
```bash
python main.py
```

**Results**
All simulation results are saved in the `simulations` directory with database files, logs, and analysis reports.

### Live Simulation Config Explorer

The project includes a modern React-based configuration explorer built with Electron for cross-platform desktop deployment:

**Hierarchical Configuration Interface:**
- **Environment**: World settings, population parameters, resource management
- **Agent Behavior**: Movement, gathering, combat, and sharing parameters
- **Learning & AI**: General and module-specific learning configurations
- **Visualization**: Display settings, animation controls, and metrics display

The interface features a complete hierarchical folder structure with collapsible sections for intuitive parameter organization and real-time configuration editing with live validation.

**Greyscale Theme**

- Professional greyscale palette via CSS variables in `src/styles/index.css`
- Leva greyscale overrides in `src/styles/leva-theme.css` with `data-theme="custom"`
- Typography: Albertus for labels (12px), JetBrains Mono for numbers (11-12px)
- Compact controls: exactly 28px height; subtle borders; monochrome focus rings
- Optional full-UI grayscale filter: `localStorage.setItem('ui:grayscale','true')`

**📋 Issue #9: Complete Leva Folder Structure - COMPLETED**
- ✅ All configuration sections organized in logical Leva folders
- ✅ Folder hierarchy matches design specification
- ✅ Folders can be collapsed/expanded
- ✅ Configuration values properly bound to folders
- ✅ No missing parameters or orphaned controls
- ✅ Comprehensive path mapping system implemented
- ✅ Complete test coverage for all functionality
- ✅ Updated documentation and Storybook stories

**Development**
```bash
# Start the development server
npm run dev

# Start Electron development mode with HMR
npm run electron:dev

# Build for production
npm run build:prod

# Package Electron app for distribution
npm run electron:pack

# Run tests
npm run test:all

# Generate documentation
npm run docs
```

**Key Scripts:**
- `npm run dev` - Start Vite development server
- `npm run dev:electron` - Start Vite with Electron environment variables
- `npm run electron:dev` - Run Electron in development mode with HMR
- `npm run build:prod` - Build optimized production bundle
- `npm run electron:pack` - Package Electron app for distribution
- `npm run test:all` - Run both Vitest and Jest test suites
- `npm run lint` - Lint TypeScript/React code
- `npm run format` - Format code with Prettier

**Features:**
- Modern React 18 + TypeScript interface
- Real-time configuration editing with Zod validation
- **Complete hierarchical folder structure** with 4 main sections and 12 sub-folders
- **Validation Display System (Issue #12)**: Inline errors/warnings with section scoping and a summary panel
- Hot Module Replacement (HMR) for rapid development
- Cross-platform packaging (Windows, macOS, Linux)
- Zustand state management with persistence
- Advanced Leva controls integration with path mapping system
- Comprehensive testing with Vitest and Jest
- TypeScript path mapping with `@/` aliases

### Validation UI and Integration

- Components:
  - `src/components/Validation/ValidationDisplay.tsx`: Show errors/warnings for exact paths or prefixed sections.
  - `src/components/Validation/ValidationSummary.tsx`: Form-level overview with counts and top issues.

- Where used:
  - `src/components/ConfigExplorer/RightPanel.tsx`: Renders `ValidationSummary` in the Validation Status section.
  - `src/components/ConfigExplorer/LeftPanel.tsx`: Renders section-scoped `ValidationDisplay` blocks.
  - `src/components/LevaControls/LevaControls.tsx`: Debounced field validation on change updates the validation store.

- Validation service:
  - `src/services/validationService.ts` integrates Zod results and classifies extra rules: capacity=error; performance/memory=warnings.

- Tests:
  - `src/components/__tests__/ValidationDisplay.test.tsx`
  - `src/components/__tests__/ValidationSummary.test.tsx`

## Documentation

For detailed documentation and advanced usage:
- [User Guide](docs/user-guide.md)
- [Developer Guide](docs/developer-guide.md)
- [IPC API Reference](docs/ipc-api.md)
- [Deployment](docs/deployment.md)
- [Monitoring & Performance](docs/monitoring.md)
- [Electron Config Explorer Architecture](docs/electron/config_explorer_architecture.md)
- [Core Architecture](docs/core_architecture.md)
- [Full Documentation Index](docs/README.md)

## Contributing

Whether you're interested in fixing bugs, adding new features, or improving documentation, your help is appreciated.

Please see [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## Design Principles

- **Single Responsibility Principle (SRP)**: A class or module should have only one reason to change, focusing on a single responsibility to reduce complexity.
- **Open-Closed Principle (OCP)**: Entities should be open for extension (e.g., via new subclasses) but closed for modification, allowing behavior addition without altering existing code.
- **Liskov Substitution Principle (LSP)**: Subclasses must be substitutable for their base classes without breaking program behavior, honoring the base class's contract.
- **Interface Segregation Principle (ISP)**: Clients should not depend on interfaces they don't use; prefer small, specific interfaces over large, general ones.
- **Dependency Inversion Principle (DIP)**: High-level modules should depend on abstractions (e.g., interfaces), not concrete implementations, to decouple components.
- **Don't Repeat Yourself (DRY)**: Avoid duplicating code or logic; centralize shared functionality to improve maintainability and reduce errors.
- **Keep It Simple, Stupid (KISS)**: Favor simple, straightforward solutions over complex ones to enhance readability and reduce bugs.
- **Composition Over Inheritance**: Prefer composing objects (e.g., via dependencies) to achieve behavior rather than relying on inheritance hierarchies, for greater flexibility.

## Support

If you encounter any issues, please check [issues page](https://github.com/Dooders/AgentFarm/issues) or open a new issue.