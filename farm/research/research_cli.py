"""Command-line interface for the research system.

Usage:
    # Create a new research project
    python research_cli.py create "My Project" --description "Study of XYZ" --tags simulation ml

    # Create an experiment in a project
    python research_cli.py experiment "My Project" "Experiment 1" --description "Testing params" --config config.yaml

    # Run an experiment
    python research_cli.py run "My Project" exp_001 --iterations 100 --steps 2000

    # Compare two experiments
    python research_cli.py compare "My Project" exp_001 exp_002 --output comparison.pdf

    # Add a literature reference
    python research_cli.py add-literature "My Project" "Paper Title" --authors "Smith J" "Jones K" --year 2023 --citation-key "smith2023" --pdf paper.pdf --notes "Important findings"

    # Add a protocol
    python research_cli.py add-protocol "My Project" "Analysis Protocol" --content-file protocol.md --category analysis

    # Export project results
    python research_cli.py export "My Project" ./output_directory

    # Update project status
    python research_cli.py status "My Project" "in_progress"

    # List experiments in a project
    python research_cli.py list-experiments "My Project"

Note:
    Run the script directly with 'python research_cli.py' rather than using 'python -m'.
    Make sure you're in the directory containing the script when running these commands.
"""

import argparse
from pathlib import Path

from farm.core.config import SimulationConfig
from research.research import ResearchProject


def main():
    parser = argparse.ArgumentParser(description="Research Project Management Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Create project
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("name")
    create_parser.add_argument("--description", default="")
    create_parser.add_argument("--tags", nargs="+", default=[])

    # Create experiment
    exp_parser = subparsers.add_parser("experiment")
    exp_parser.add_argument("project")
    exp_parser.add_argument("name")
    exp_parser.add_argument("--description", default="")
    exp_parser.add_argument("--config", required=True)

    # Run experiment
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("project")
    run_parser.add_argument("experiment_id")
    run_parser.add_argument("--iterations", type=int, default=10)
    run_parser.add_argument("--steps", type=int, default=1000)

    # Compare experiments
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("project")
    compare_parser.add_argument("exp1")
    compare_parser.add_argument("exp2")
    compare_parser.add_argument("--output")

    # Add literature reference
    lit_parser = subparsers.add_parser("add-literature")
    lit_parser.add_argument("project")
    lit_parser.add_argument("title")
    lit_parser.add_argument("--authors", nargs="+", required=True)
    lit_parser.add_argument("--year", type=int, required=True)
    lit_parser.add_argument("--citation-key", required=True)
    lit_parser.add_argument("--pdf")
    lit_parser.add_argument("--notes")

    # Add protocol
    protocol_parser = subparsers.add_parser("add-protocol")
    protocol_parser.add_argument("project")
    protocol_parser.add_argument("name")
    protocol_parser.add_argument("--content-file", required=True)
    protocol_parser.add_argument("--category", default="analysis")

    # Export results
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("project")
    export_parser.add_argument("output_dir")

    # Update status
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("project")
    status_parser.add_argument("new_status")

    # List experiments
    list_exp_parser = subparsers.add_parser("list-experiments")
    list_exp_parser.add_argument("project")

    args = parser.parse_args()

    if args.command == "create":
        project = ResearchProject(args.name, args.description, tags=args.tags)
        print(f"Created research project: {args.name}")

    elif args.command == "experiment":
        project = ResearchProject(args.project)
        try:
            config = SimulationConfig.from_yaml(args.config)
            exp_path = project.create_experiment(args.name, args.description, config)
            exp_id = exp_path.name  # Get the folder name as the experiment ID
            print(f"Created experiment with ID: {exp_id}")
            print(f"Use this ID when running the experiment")
        except TypeError as e:
            print(f"Error: Could not save configuration - {str(e)}")
            print("Make sure your config file only contains JSON-serializable values")
            exit(1)
        except Exception as e:
            print(f"Error creating experiment: {str(e)}")
            exit(1)

    elif args.command == "list-experiments":
        project = ResearchProject(args.project)
        experiments = project.list_experiments()
        if experiments:
            print("\nAvailable experiments:")
            for exp_id in experiments:
                print(f"  {exp_id}")
        else:
            print("No experiments found in this project")

    elif args.command == "run":
        project = ResearchProject(args.project)
        try:
            project.run_experiment(args.experiment_id, args.iterations, args.steps)
        except ValueError as e:
            print(f"Error: {str(e)}")
            print("Use 'list-experiments' command to see available experiment IDs")
            exit(1)

    elif args.command == "compare":
        project = ResearchProject(args.project)
        output = Path(args.output) if args.output else None
        results = project.compare_experiments(args.exp1, args.exp2, output)
        if not args.output:
            print(results)

    elif args.command == "add-literature":
        project = ResearchProject(args.project)
        pdf_path = Path(args.pdf) if args.pdf else None
        project.add_literature_reference(
            args.title,
            args.authors,
            args.year,
            args.citation_key,
            pdf_path,
            args.notes or "",
        )
        print(f"Added literature reference: {args.citation_key}")

    elif args.command == "add-protocol":
        project = ResearchProject(args.project)
        with open(args.content_file) as f:
            content = f.read()
        project.add_protocol(args.name, content, args.category)
        print(f"Added protocol: {args.name}")

    elif args.command == "export":
        project = ResearchProject(args.project)
        project.export_results(Path(args.output_dir))
        print(f"Exported results to: {args.output_dir}")

    elif args.command == "status":
        project = ResearchProject(args.project)
        project.update_status(args.new_status)
        print(f"Updated project status to: {args.new_status}")


if __name__ == "__main__":
    main()
