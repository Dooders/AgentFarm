# Research System Definitions

## Core Concepts
- **Research Project**: A structured investigation consisting of multiple experiments designed to address a specific research question or hypothesis
- **Experiment**: A controlled set of simulations with defined parameters and variables to test specific aspects of the research question
- **Simulation**: A discrete computational model that executes a sequence of steps, involving agents interacting within an arena
- **Agent**: An autonomous entity with defined behaviors, capabilities, and decision-making processes
- **Arena**: The bounded environment providing physical laws, constraints, and interaction rules for agents

## Data Architecture
- **State**: A complete snapshot of the system at a given point in time
- **Event**: A discrete occurrence or interaction between agents or environment
- **Metric**: A quantifiable measurement of system behavior or agent performance
- **Analysis**: Processed data and interpretations derived from raw simulation results

## Folder Structure
research/
├── {research_name}/
│   ├── metadata.json           # Research project configuration and metadata
│   ├── hypothesis.md           # Research questions and hypotheses
│   ├── literature/             # Related research and references
│   │   ├── papers/            # Referenced academic papers
│   │   └── bibliography.bib   # Bibliography in BibTeX format
│   ├── protocols/             # Standard procedures and methodologies
│   │   ├── validation.md      # Validation protocols
│   │   └── analysis.md        # Analysis procedures
│   └── experiments/
│       └── {experiment_name}/
│           ├── experiment_config.json    # Experiment parameters and setup
│           ├── experiment_design.md      # Detailed experiment methodology
│           ├── pilot_results/           # Initial test runs and calibration data
│           ├── simulations/
│           │   └── {simulation_id}/
│           │       ├── data/
│           │       │   ├── simulation.db     # Time-series data of simulation steps
│           │       │   ├── analysis.db       # Processed results and metrics
│           │       │   ├── states.farm       # System state snapshots
│           │       │   ├── config.json       # Simulation-specific configuration
│           │       │   ├── simulation.log    # Detailed execution log
│           │       │   └── raw_outputs/      # Raw output files before processing
│           │       └── analysis/
│           │           ├── charts/
│           │           │   ├── metrics/      # Standard metric visualizations
│           │           │   └── custom/       # Special-purpose visualizations
│           │           └── reports/
│           │               ├── chart_analysis.txt         # AI interpretation of charts
│           │               ├── validation_results.txt     # Simulation validation report
│           │               ├── simulation_summary.txt     # Contextual analysis
│           │               └── anomalies.txt             # Detected anomalies and edge cases
│           ├── aggregate_analysis/
│           │   ├── comparative_results/      # Cross-simulation analysis
│           │   ├── statistical_tests/        # Statistical validation results
│           │   ├── experiment_summary.md     # Overall experiment findings
│           │   └── reproducibility/          # Reproducibility verification data
│           └── artifacts/
│               ├── presentations/            # Experiment presentations and demos
│               ├── notebooks/                # Jupyter notebooks for analysis
│               └── media/                    # Videos, animations, key visualizations
│               └── benchmarks/   # Performance benchmarks
│               └── reviews/      # Peer review feedback

## Additional Data Types
- **Validation Data**: Reference datasets and expected outcomes for validation
- **Statistical Tests**: Results of statistical significance testing
- **Anomalies**: Unexpected behaviors or edge cases detected
- **Reproducibility Data**: Information needed to reproduce results
- **Pilot Data**: Initial test runs used for calibration
- **Literature References**: Related papers and citations
- **Protocols**: Standard procedures for validation and analysis