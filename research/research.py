"""
Research System Implementation

This module implements a structured research management system for organizing and conducting
simulation experiments. It follows the architecture defined in docs/research.md.

The system provides functionality for:
- Creating and managing research projects
- Organizing experiments and their results
- Tracking literature references
- Managing research protocols
- Comparing experimental results
- Exporting research artifacts

Example Usage
------------
>>> from research import ResearchProject
>>> project = ResearchProject("agent_behavior", "Study of emergent behaviors")
>>> project.create_experiment("baseline", "Baseline behavior patterns", config)
>>> project.run_experiment("baseline_20230615", iterations=100)
>>> project.export_results(Path("./results"))

Classes
-------
ResearchMetadata : dataclass
    Stores metadata about a research project
ResearchProject : class
    Main class for managing research projects and experiments
"""

import errno
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from farm.core.config import SimulationConfig
from farm.core.experiment_runner import ExperimentRunner


@dataclass
class ResearchMetadata:
    """
    Research project metadata container.

    Attributes
    ----------
    name : str
        Name of the research project
    description : str
        Description of research goals and context
    created_at : str
        ISO format timestamp of project creation
    updated_at : str
        ISO format timestamp of last update
    status : str
        Current project status (e.g., "initialized", "in_progress", "completed")
    tags : List[str]
        List of tags for categorizing the research
    version : str, optional
        Version string, defaults to "1.0.0"
    """

    name: str
    description: str
    created_at: str
    updated_at: str
    status: str
    tags: List[str]
    version: str = "1.0.0"


class ResearchProject:
    """
    Manages a research project including experiments, data, and analysis.

    This class provides methods for organizing research artifacts, running experiments,
    and managing results according to the defined research system architecture.

    Attributes
    ----------
    name : str
        Name of the research project
    description : str
        Description of research goals and context
    base_path : Path
        Base directory for all research projects
    project_path : Path
        Directory for this specific project
    metadata : ResearchMetadata
        Project metadata container
    logger : logging.Logger
        Project-specific logger

    Methods
    -------
    create_experiment(name, description, config)
        Create a new experiment with given configuration
    run_experiment(experiment_id, iterations, steps_per_iteration)
        Run an experiment with specified parameters
    add_literature_reference(title, authors, year, citation_key, pdf_path, notes)
        Add a literature reference to the project bibliography
    add_protocol(name, content, category)
        Add a research protocol document
    compare_experiments(exp_id_1, exp_id_2, output_path)
        Compare results between two experiments
    export_results(output_dir)
        Export research results and artifacts
    update_status(status)
        Update project status
    list_experiments()
        Return a list of experiment IDs in this project
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        base_path: Union[str, Path] = "results",
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize a research project.

        Parameters
        ----------
        name : str
            Name of the research project
        description : str, optional
            Description of research goals and context
        base_path : Union[str, Path], optional
            Base directory for research projects, defaults to "research"
        tags : Optional[List[str]], optional
            Tags for categorizing the research
        """
        self.name = name
        self.description = description
        self.tags = tags or []

        # Setup directory structure
        self.base_path = Path(base_path)
        self.project_path = self.base_path / name

        # Delete existing project directory if it exists
        if self.project_path.exists():
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    shutil.rmtree(self.project_path)
                    break
                except PermissionError as e:
                    if attempt == max_attempts - 1:  # Last attempt
                        raise Exception(
                            f"Unable to delete directory after {max_attempts} attempts: {e}"
                        )
                    time.sleep(1)  # Wait a second before retrying

        self._setup_directory_structure()

        # Initialize metadata
        self.metadata = ResearchMetadata(
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status="initialized",
            tags=self.tags,
        )

        # Save initial metadata
        self._save_metadata()

        # Setup logging
        self.logger = self._setup_logging()

    def _setup_directory_structure(self) -> None:
        """Create the research project directory structure."""
        # Create main project directory
        self.project_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories based on research.md specification
        directories = [
            "literature/papers",
            "protocols",
            "experiments",
            "experiments/pilot-results",
            "experiments/simulations",
            "experiments/aggregate-analysis",
            "experiments/artifacts/presentations",
            "experiments/artifacts/notebooks",
            "experiments/artifacts/media",
            "experiments/artifacts/benchmarks",
            "experiments/artifacts/reviews",
        ]

        for directory in directories:
            (self.project_path / directory).mkdir(parents=True, exist_ok=True)

        # Create initial files
        self._create_initial_files()

    def _create_initial_files(self) -> None:
        """Create initial project files."""
        # Create hypothesis.md
        hypothesis_path = self.project_path / "hypothesis.md"
        if not hypothesis_path.exists():
            with open(hypothesis_path, "w") as f:
                f.write(
                    "# Research Hypotheses\n\n## Primary Hypothesis\n\n## Secondary Hypotheses\n"
                )

        # Create bibliography file
        bib_path = self.project_path / "literature" / "bibliography.bib"
        if not bib_path.exists():
            bib_path.touch()

        # Create protocol files
        for protocol in ["validation.md", "analysis.md"]:
            protocol_path = self.project_path / "protocols" / protocol
            if not protocol_path.exists():
                protocol_path.touch()

    def _setup_logging(self) -> logging.Logger:
        """Configure project-specific logging."""
        log_path = self.project_path / "research.log"

        logger = logging.getLogger(f"research.{self.name}")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _save_metadata(self) -> None:
        """Save project metadata to JSON file."""
        metadata_path = self.project_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(vars(self.metadata), f, indent=2)

    def create_experiment(
        self,
        name: str,
        description: str,
        config: SimulationConfig,
        path: Optional[Path] = None,
    ) -> str:
        """
        Create a new experiment within the research project.

        Creates a new experiment directory with configuration and design documents.
        The experiment ID is generated using the name and current timestamp.

        Parameters
        ----------
        name : str
            Name of the experiment
        description : str
            Description of experiment purpose
        config : SimulationConfig
            Base configuration for the experiment

        Returns
        -------
        str
            Path to the created experiment directory

        Examples
        --------
        >>> config = SimulationConfig.from_yaml("configs/base.yaml")
        >>> project.create_experiment("baseline", "Baseline behavior", config)
        'results/project_name/experiments/simulations/baseline_20230615_120000'
        """
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{name}_{timestamp}"
        exp_path = self.project_path / "experiments" / "data" / exp_id
        exp_path.mkdir(parents=True, exist_ok=True)

        # Save experiment configuration
        config_path = exp_path / "experiment-config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Create experiment design document
        design_path = exp_path / "experiment-design.md"
        with open(design_path, "w") as f:
            f.write(f"# {name}\n\n## Description\n{description}\n\n## Methodology\n")

        self.logger.info(f"Created experiment: {exp_id}")
        return str(exp_path)

    def run_experiment(
        self, experiment_id: str, iterations: int = 10, steps_per_iteration: int = 1000
    ) -> None:
        """
        Run an experiment with the specified parameters.

        Executes multiple iterations of the experiment using the stored configuration.
        Results are automatically saved and analyzed.

        Parameters
        ----------
        experiment_id : str
            ID of the experiment to run
        iterations : int, optional
            Number of iterations to run, defaults to 10
        steps_per_iteration : int, optional
            Number of steps per iteration, defaults to 1000

        Raises
        ------
        ValueError
            If the experiment ID is not found
        Exception
            If there is an error during experiment execution
        """
        exp_path = self.project_path / "experiments" / experiment_id
        if not exp_path.exists():
            raise ValueError(f"Experiment {experiment_id} not found")

        # Load experiment config
        config_path = exp_path / "experiment-config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        config = SimulationConfig.from_dict(config_dict)

        # Create experiment runner
        runner = ExperimentRunner(config, experiment_id)

        try:
            # Run experiment
            runner.run_iterations(iterations, num_steps=steps_per_iteration)

            # Generate report
            runner.generate_report()

            self.logger.info(f"Completed experiment: {experiment_id}")

        except Exception as e:
            self.logger.error(f"Error running experiment {experiment_id}: {str(e)}")
            raise

    def add_literature_reference(
        self,
        title: str,
        authors: List[str],
        year: int,
        citation_key: str,
        pdf_path: Optional[Path] = None,
        notes: str = "",
    ) -> None:
        """
        Add a literature reference to the project bibliography.

        Creates a BibTeX entry and optionally stores the PDF and notes.

        Parameters
        ----------
        title : str
            Paper title
        authors : List[str]
            List of author names
        year : int
            Publication year
        citation_key : str
            BibTeX citation key
        pdf_path : Optional[Path], optional
            Path to PDF file to copy into papers directory
        notes : str, optional
            Research notes about the paper

        Examples
        --------
        >>> project.add_literature_reference(
        ...     "Example Paper",
        ...     ["Author One", "Author Two"],
        ...     2023,
        ...     "example2023",
        ...     Path("papers/example.pdf"),
        ...     "Important findings about X"
        ... )
        """
        # Create BibTeX entry
        bib_entry = (
            f"@article{{{citation_key},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{' and '.join(authors)}}},\n"
            f"  year = {{{year}}}\n"
            "}}\n\n"
        )

        # Append to bibliography
        bib_path = self.project_path / "literature" / "bibliography.bib"
        with open(bib_path, "a") as f:
            f.write(bib_entry)

        # Copy PDF if provided
        if pdf_path:
            papers_dir = self.project_path / "literature" / "papers"
            shutil.copy2(pdf_path, papers_dir / f"{citation_key}.pdf")

        # Save notes if provided
        if notes:
            notes_path = (
                self.project_path / "literature" / "papers" / f"{citation_key}_notes.md"
            )
            with open(notes_path, "w") as f:
                f.write(f"# Notes on {title}\n\n{notes}")

        self.logger.info(f"Added literature reference: {citation_key}")

    def add_protocol(self, name: str, content: str, category: str = "analysis") -> None:
        """
        Add a research protocol document.

        Parameters
        ----------
        name : str
            Protocol name
        content : str
            Protocol content in markdown format
        category : str
            Protocol category (analysis, validation, etc)
        """
        protocol_path = self.project_path / "protocols" / f"{name}.md"
        with open(protocol_path, "w") as f:
            f.write(content)

        self.logger.info(f"Added {category} protocol: {name}")

    def compare_experiments(
        self, exp_id_1: str, exp_id_2: str, output_path: Optional[Path] = None
    ) -> Dict:
        """
        Compare results between two experiments.

        Parameters
        ----------
        exp_id_1 : str
            First experiment ID
        exp_id_2 : str
            Second experiment ID
        output_path : Optional[Path]
            Path to save comparison report

        Returns
        -------
        Dict
            Comparison results
        """
        from farm.tools.compare_sims import compare_simulations

        exp1_path = self.project_path / "experiments" / exp_id_1
        exp2_path = self.project_path / "experiments" / exp_id_2

        if not exp1_path.exists() or not exp2_path.exists():
            raise ValueError("One or both experiments not found")

        comparison = compare_simulations(exp1_path, exp2_path)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(comparison, f, indent=2)

        return comparison

    def export_results(self, output_dir: Path) -> None:
        """
        Export research results and artifacts to a directory.

        Parameters
        ----------
        output_dir : Path
            Directory to export results to
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy key files and directories
        dirs_to_copy = [
            "experiments/aggregate-analysis",
            "experiments/artifacts",
            "protocols",
            "hypothesis.md",
            "metadata.json",
        ]

        for item in dirs_to_copy:
            src = self.project_path / item
            dst = output_dir / item
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                shutil.copytree(src, dst)

        self.logger.info(f"Exported research results to {output_dir}")

    def update_status(self, status: str) -> None:
        """Update research project status."""
        self.metadata.status = status
        self.metadata.updated_at = datetime.now().isoformat()
        self._save_metadata()
        self.logger.info(f"Updated project status to: {status}")

    def list_experiments(self) -> List[str]:
        """Return a list of experiment IDs in this project."""
        exp_dir = self.project_path / "experiments"
        if not exp_dir.exists():
            return []
        return [d.name for d in exp_dir.iterdir() if d.is_dir()]
