# AgentFarm Research System

![Project Status](https://img.shields.io/badge/status-in%20development-orange)

A structured research management system for AgentFarm, designed to organize, track, and analyze simulation experiments systematically.

## Quick Start

1. **Create a new research project with template:**
```bash
python create_research_template.py my_research \
    --description "Investigation of emergent behaviors" \
    --tags emergence behavior adaptation
```

2. **Create an experiment:**
```bash
python research_cli.py experiment my_research baseline \
    --config configs/base.yaml \
    --description "Baseline behavior patterns"
```

3. **Run the experiment:**
```bash
python research_cli.py run my_research baseline_20230615_120000 \
    --iterations 100 \
    --steps 1000
```

4. **Compare experiments:**
```bash
python research_cli.py compare my_research \
    exp1_20230615_120000 exp2_20230615_130000 \
    --output comparison.json
```

## Features

### Research Organization
- Structured project hierarchy
- Experiment tracking and versioning
- Literature reference management
- Protocol documentation
- Results organization and export

### Experiment Management
- Configuration version control
- Automated experiment execution
- Comparative analysis tools
- Result validation protocols

### Documentation
- Hypothesis tracking
- Methodology documentation
- Analysis protocols
- Result summaries

## Directory Structure

```
research/
├── {research-name}/
│   ├── metadata.json            # Project configuration and metadata
│   ├── hypothesis.md           # Research questions and hypotheses
│   ├── literature/            # Related research and references
│   ├── protocols/             # Standard procedures
│   └── experiments/           # Experiment data and results
```

## Command Reference

### Project Management
```bash
# Create new project
python research_cli.py create project_name --description "..." --tags tag1 tag2

# Update project status
python research_cli.py status project_name "in_progress"
```

### Literature Management
```bash
# Add reference
python research_cli.py add-literature project_name \
    "Paper Title" \
    --authors "Author One" "Author Two" \
    --year 2023 \
    --citation-key paper2023 \
    --pdf path/to/paper.pdf \
    --notes "Key findings..."
```

### Protocol Management
```bash
# Add protocol
python research_cli.py add-protocol project_name \
    protocol_name \
    --content-file protocol.md \
    --category analysis
```

### Results Management
```bash
# Export results
python research_cli.py export project_name ./exported_results
```

## Integration with AgentFarm

The research system is fully integrated with AgentFarm's simulation capabilities:
- Direct access to simulation configurations
- Automated experiment execution
- Built-in analysis tools
- Result visualization and comparison

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create your first research project:
```bash
python create_research_template.py my_first_research
```

3. Follow the printed instructions to:
   - Edit hypothesis.md
   - Customize protocols
   - Add literature references
   - Create your first experiment

## Documentation

For detailed system definitions and architecture, see:
- [Research System Definitions](docs/research.md)
- [AgentFarm Documentation](docs/README.md)