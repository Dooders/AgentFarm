"""
Database validation module for simulation data integrity and consistency checks.

This module provides comprehensive validation of simulation databases to ensure
data integrity, referential consistency, and statistical validity. It includes
both basic integrity checks and advanced statistical validators.

Key Components:
- ValidationResult: Individual validation check results
- ValidationReport: Aggregated validation results with reporting
- IntegrityValidators: Basic database integrity checks
- StatisticalValidators: Advanced consistency and range checks
- validate_simulation_database: Main entry point for validation

Usage:
    from farm.database.validation import validate_simulation_database

    report = validate_simulation_database("simulation.db")
    if report.has_errors():
        print(f"Found {report.error_count} errors")
    print(report.get_summary())
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from farm.database.models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    HealthIncident,
    ResourceModel,
    Simulation,
    SimulationConfig,
    SimulationStepModel,
)
from farm.database.utils import extract_agent_counts_from_json
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    violation_count: int = 0

    def __post_init__(self):
        """Ensure violation_count is set correctly."""
        if self.violation_count == 0 and not self.passed:
            self.violation_count = 1


@dataclass
class ValidationReport:
    """Aggregated validation results with reporting capabilities."""

    results: List[ValidationResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    database_path: Optional[str] = None
    simulation_id: Optional[str] = None

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.results.append(result)

    def finalize(self) -> None:
        """Mark the validation as complete."""
        self.end_time = time.time()

    @property
    def total_checks(self) -> int:
        """Total number of validation checks performed."""
        return len(self.results)

    @property
    def passed_checks(self) -> int:
        """Number of checks that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_checks(self) -> int:
        """Number of checks that failed."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def error_count(self) -> int:
        """Number of error-level issues."""
        return sum(1 for r in self.results if r.severity == ValidationSeverity.ERROR and not r.passed)

    @property
    def warning_count(self) -> int:
        """Number of warning-level issues."""
        return sum(1 for r in self.results if r.severity == ValidationSeverity.WARNING and not r.passed)

    @property
    def info_count(self) -> int:
        """Number of info-level issues."""
        return sum(1 for r in self.results if r.severity == ValidationSeverity.INFO and not r.passed)

    @property
    def total_violations(self) -> int:
        """Total number of violations across all checks."""
        return sum(r.violation_count for r in self.results if not r.passed)

    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return self.error_count > 0

    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return self.warning_count > 0

    def is_clean(self) -> bool:
        """Check if validation passed without any issues."""
        return self.failed_checks == 0

    def get_summary(self) -> str:
        """Get a human-readable summary of validation results."""
        if self.end_time is None:
            self.finalize()

        duration = self.end_time - self.start_time

        summary = [
            "Database Validation Summary",
            f"Database: {self.database_path or 'Unknown'}",
            f"Simulation ID: {self.simulation_id or 'Unknown'}",
            f"Duration: {duration:.2f} seconds",
            "",
            "Results:",
            f"  Total Checks: {self.total_checks}",
            f"  Passed: {self.passed_checks}",
            f"  Failed: {self.failed_checks}",
            f"  Errors: {self.error_count}",
            f"  Warnings: {self.warning_count}",
            f"  Info: {self.info_count}",
            f"  Total Violations: {self.total_violations}",
        ]

        if self.failed_checks > 0:
            summary.append("")
            summary.append("Failed Checks:")
            for result in self.results:
                if not result.passed:
                    summary.append(f"  [{result.severity.value.upper()}] {result.check_name}: {result.message}")
                    if result.violation_count > 1:
                        summary.append(f"    Violations: {result.violation_count}")

        return "\n".join(summary)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        if self.end_time is None:
            self.finalize()

        return {
            "database_path": self.database_path,
            "simulation_id": self.simulation_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time,
            "summary": {
                "total_checks": self.total_checks,
                "passed_checks": self.passed_checks,
                "failed_checks": self.failed_checks,
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "info_count": self.info_count,
                "total_violations": self.total_violations,
            },
            "results": [
                {
                    "check_name": r.check_name,
                    "severity": r.severity.value,
                    "passed": r.passed,
                    "message": r.message,
                    "violation_count": r.violation_count,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


class IntegrityValidator:
    """Validates basic database integrity constraints."""

    def __init__(self, session, report: ValidationReport):
        self.session = session
        self.report = report

    def validate_table_existence(self) -> None:
        """Check that all required tables exist."""
        required_tables = {
            "agents",
            "agent_states",
            "simulation_steps",
            "agent_actions",
            "resource_states",
            "health_incidents",
            "simulation_config",
            "simulations",
        }

        inspector = inspect(self.session.bind)
        existing_tables = set(inspector.get_table_names())
        missing_tables = required_tables - existing_tables

        if missing_tables:
            self.report.add_result(
                ValidationResult(
                    check_name="table_existence",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Missing required tables: {', '.join(missing_tables)}",
                    violation_count=len(missing_tables),
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="table_existence",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All required tables present",
                )
            )

    def validate_primary_keys(self) -> None:
        """Check primary key uniqueness and constraints."""
        # Check agents table
        duplicate_agents = self.session.execute(
            text("""
            SELECT agent_id, COUNT(*) as count 
            FROM agents 
            GROUP BY agent_id 
            HAVING COUNT(*) > 1
        """)
        ).fetchall()

        if duplicate_agents:
            self.report.add_result(
                ValidationResult(
                    check_name="agents_primary_key",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Duplicate agent IDs found: {len(duplicate_agents)}",
                    violation_count=len(duplicate_agents),
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="agents_primary_key",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="No duplicate agent IDs",
                )
            )

        # Check agent_states composite primary key
        duplicate_states = self.session.execute(
            text("""
            SELECT id, COUNT(*) as count 
            FROM agent_states 
            GROUP BY id 
            HAVING COUNT(*) > 1
        """)
        ).fetchall()

        if duplicate_states:
            self.report.add_result(
                ValidationResult(
                    check_name="agent_states_primary_key",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Duplicate agent state IDs found: {len(duplicate_states)}",
                    violation_count=len(duplicate_states),
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="agent_states_primary_key",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="No duplicate agent state IDs",
                )
            )

    def validate_foreign_keys(self) -> None:
        """Check foreign key referential integrity."""
        # Check agent_states -> agents
        orphaned_states = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agent_states ast 
            LEFT JOIN agents a ON ast.agent_id = a.agent_id 
            WHERE a.agent_id IS NULL
        """)
        ).scalar()

        if orphaned_states > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="agent_states_foreign_key",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Agent states with invalid agent_id: {orphaned_states}",
                    violation_count=orphaned_states,
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="agent_states_foreign_key",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All agent states have valid agent references",
                )
            )

        # Check actions -> agents
        orphaned_actions = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agent_actions aa 
            LEFT JOIN agents a ON aa.agent_id = a.agent_id 
            WHERE a.agent_id IS NULL
        """)
        ).scalar()

        if orphaned_actions > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="actions_foreign_key",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Actions with invalid agent_id: {orphaned_actions}",
                    violation_count=orphaned_actions,
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="actions_foreign_key",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All actions have valid agent references",
                )
            )

    def validate_not_null_constraints(self) -> None:
        """Check NOT NULL constraint violations."""
        # Check agents table
        null_agent_ids = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agents WHERE agent_id IS NULL
        """)
        ).scalar()

        if null_agent_ids > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="agents_not_null",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Agents with NULL agent_id: {null_agent_ids}",
                    violation_count=null_agent_ids,
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="agents_not_null",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="No NULL agent IDs",
                )
            )

        # Check agent_states table
        null_state_ids = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agent_states WHERE id IS NULL
        """)
        ).scalar()

        if null_state_ids > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="agent_states_not_null",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Agent states with NULL id: {null_state_ids}",
                    violation_count=null_state_ids,
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="agent_states_not_null",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="No NULL agent state IDs",
                )
            )

    def validate_temporal_consistency(self) -> None:
        """Check temporal consistency (birth_time < death_time, step ordering)."""
        # Check birth_time < death_time
        invalid_lifespans = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agents 
            WHERE death_time IS NOT NULL AND death_time <= birth_time
        """)
        ).scalar()

        if invalid_lifespans > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="temporal_consistency",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Agents with death_time <= birth_time: {invalid_lifespans}",
                    violation_count=invalid_lifespans,
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="temporal_consistency",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All agent lifespans are temporally consistent",
                )
            )

        # Check step ordering in agent_states
        invalid_steps = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agent_states 
            WHERE step_number < 0
        """)
        ).scalar()

        if invalid_steps > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="step_ordering",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Agent states with negative step numbers: {invalid_steps}",
                    violation_count=invalid_steps,
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="step_ordering",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All step numbers are non-negative",
                )
            )


class StatisticalValidator:
    """Validates statistical consistency and data ranges."""

    def __init__(self, session, report: ValidationReport):
        self.session = session
        self.report = report

    def validate_health_ranges(self) -> None:
        """Check that health values are within valid ranges."""
        # Check current_health >= 0 and <= starting_health
        invalid_health = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agent_states 
            WHERE current_health < 0 OR current_health > starting_health
        """)
        ).scalar()

        if invalid_health > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="health_ranges",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Agent states with invalid health values: {invalid_health}",
                    violation_count=invalid_health,
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="health_ranges",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All health values are within valid ranges",
                )
            )

    def validate_resource_ranges(self) -> None:
        """Check that resource values are non-negative."""
        # Check resource_level >= 0
        negative_resources = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agent_states 
            WHERE resource_level < 0
        """)
        ).scalar()

        if negative_resources > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="resource_ranges",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Agent states with negative resources: {negative_resources}",
                    violation_count=negative_resources,
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="resource_ranges",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All resource values are non-negative",
                )
            )

        # Check resource amounts in resource_states
        negative_resource_amounts = self.session.execute(
            text("""
            SELECT COUNT(*) FROM resource_states 
            WHERE amount < 0
        """)
        ).scalar()

        if negative_resource_amounts > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="resource_amounts",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Resource states with negative amounts: {negative_resource_amounts}",
                    violation_count=negative_resource_amounts,
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="resource_amounts",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All resource amounts are non-negative",
                )
            )

    def validate_generation_monotonicity(self) -> None:
        """Check that offspring generations are parent + 1."""
        # Reconstruct from agents table - check offspring generations match parent + 1
        from farm.database.data_types import GenomeId
        
        invalid_count = 0
        offspring_agents = self.session.query(AgentModel).filter(AgentModel.birth_time > 0).all()
        
        for offspring in offspring_agents:
            try:
                genome = GenomeId.from_string(offspring.genome_id)
                if genome.parent_ids:
                    parent_id = genome.parent_ids[0]
                    parent = (
                        self.session.query(AgentModel)
                        .filter(AgentModel.agent_id == parent_id)
                        .first()
                    )
                    if parent and offspring.generation != parent.generation + 1:
                        invalid_count += 1
            except Exception as e:
                logger.error(
                    f"Error parsing genome ID or parent lookup for agent_id={offspring.agent_id}, genome_id={offspring.genome_id}: {e}"
                )
                continue
        
        if invalid_count > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="generation_monotonicity",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Found {invalid_count} offspring with invalid generation (not parent + 1)",
                    violation_count=invalid_count,
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="generation_monotonicity",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All offspring have valid generation numbers (parent + 1)",
                )
            )

    def validate_population_consistency(self) -> None:
        """Check that population counts in simulation_steps match actual agent counts."""
        # Get the latest step
        latest_step = self.session.execute(
            text("""
            SELECT MAX(step_number) FROM simulation_steps
        """)
        ).scalar()

        if latest_step is None:
            self.report.add_result(
                ValidationResult(
                    check_name="population_consistency",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="No simulation steps found - database is empty",
                )
            )
            return

        # Get population counts from simulation_steps (using JSON column)
        step_row = self.session.execute(
            text("""
            SELECT total_agents, agent_type_counts
            FROM simulation_steps 
            WHERE step_number = :step
        """),
            {"step": latest_step},
        ).fetchone()
        
        if step_row is None:
            self.report.add_result(
                ValidationResult(
                    check_name="population_consistency",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"No simulation step data for step {latest_step}",
                )
            )
            return
        
        agent_type_counts = extract_agent_counts_from_json(step_row[1])
        step_counts = (
            step_row[0],  # total_agents
            agent_type_counts.get("system", 0),
            agent_type_counts.get("independent", 0),
            agent_type_counts.get("control", 0),
        )

        # Get actual agent counts
        actual_total = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agents WHERE death_time IS NULL OR death_time > :step
        """),
            {"step": latest_step},
        ).scalar()

        actual_system = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agents 
            WHERE agent_type = 'system' AND (death_time IS NULL OR death_time > :step)
        """),
            {"step": latest_step},
        ).scalar()

        actual_independent = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agents 
            WHERE agent_type = 'independent' AND (death_time IS NULL OR death_time > :step)
        """),
            {"step": latest_step},
        ).scalar()

        actual_control = self.session.execute(
            text("""
            SELECT COUNT(*) FROM agents 
            WHERE agent_type = 'control' AND (death_time IS NULL OR death_time > :step)
        """),
            {"step": latest_step},
        ).scalar()

        violations = 0
        issues = []

        # Allow small differences due to timing issues between metrics calculation and database commits
        # This is a known issue where new agents from reproduction might not be fully committed
        # to the database when metrics are calculated, causing 1-2 agent differences
        tolerance = 2

        if abs(step_counts[0] - actual_total) > tolerance:
            violations += 1
            issues.append(f"Total agents: step={step_counts[0]}, actual={actual_total}")

        if abs(step_counts[1] - actual_system) > tolerance:
            violations += 1
            issues.append(f"System agents: step={step_counts[1]}, actual={actual_system}")

        if abs(step_counts[2] - actual_independent) > tolerance:
            violations += 1
            issues.append(f"Independent agents: step={step_counts[2]}, actual={actual_independent}")

        if abs(step_counts[3] - actual_control) > tolerance:
            violations += 1
            issues.append(f"Control agents: step={step_counts[3]}, actual={actual_control}")

        if violations > 0:
            self.report.add_result(
                ValidationResult(
                    check_name="population_consistency",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Population count mismatches: {violations}",
                    violation_count=violations,
                    details={"issues": issues},
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="population_consistency",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="Population counts are consistent",
                )
            )

    def validate_step_continuity(self) -> None:
        """Check that simulation steps are continuous."""
        # Get step numbers
        steps = self.session.execute(
            text("""
            SELECT DISTINCT step_number FROM simulation_steps 
            ORDER BY step_number
        """)
        ).fetchall()

        if not steps:
            self.report.add_result(
                ValidationResult(
                    check_name="step_continuity",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="No simulation steps found - database is empty",
                )
            )
            return

        step_numbers = [row[0] for row in steps]
        expected_steps = list(range(min(step_numbers), max(step_numbers) + 1))
        missing_steps = set(expected_steps) - set(step_numbers)

        if missing_steps:
            self.report.add_result(
                ValidationResult(
                    check_name="step_continuity",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Missing simulation steps: {len(missing_steps)}",
                    violation_count=len(missing_steps),
                    details={"missing_steps": sorted(missing_steps)},
                )
            )
        else:
            self.report.add_result(
                ValidationResult(
                    check_name="step_continuity",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All simulation steps are continuous",
                )
            )


def validate_simulation_database(
    database_path: str,
    simulation_id: Optional[str] = None,
    include_integrity: bool = True,
    include_statistical: bool = True,
) -> ValidationReport:
    """
    Validate a simulation database for integrity and consistency.

    Parameters
    ----------
    database_path : str
        Path to the SQLite database file
    simulation_id : Optional[str]
        Simulation ID to validate (if None, validates all data)
    include_integrity : bool
        Whether to run integrity validators
    include_statistical : bool
        Whether to run statistical validators

    Returns
    -------
    ValidationReport
        Comprehensive validation report with all results
    """
    report = ValidationReport(database_path=database_path, simulation_id=simulation_id)

    try:
        # Create database connection
        engine = create_engine(f"sqlite:///{database_path}")
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Run integrity validators
            if include_integrity:
                integrity_validator = IntegrityValidator(session, report)
                integrity_validator.validate_table_existence()
                integrity_validator.validate_primary_keys()
                integrity_validator.validate_foreign_keys()
                integrity_validator.validate_not_null_constraints()
                integrity_validator.validate_temporal_consistency()

            # Run statistical validators
            if include_statistical:
                statistical_validator = StatisticalValidator(session, report)
                statistical_validator.validate_health_ranges()
                statistical_validator.validate_resource_ranges()
                statistical_validator.validate_generation_monotonicity()
                statistical_validator.validate_population_consistency()
                statistical_validator.validate_step_continuity()

        finally:
            session.close()
            engine.dispose()

    except Exception as e:
        logger.error(f"Database validation failed: {e}")
        report.add_result(
            ValidationResult(
                check_name="database_connection",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Failed to connect to database: {str(e)}",
            )
        )

    finally:
        report.finalize()

    return report
