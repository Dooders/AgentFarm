"""
Tests for database validation module.

This module tests the database validation functionality including:
- ValidationResult and ValidationReport classes
- IntegrityValidator functionality
- StatisticalValidator functionality
- Main validation function
- CLI tool functionality
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from farm.database.models import Base
from farm.database.validation import (
    IntegrityValidator,
    StatisticalValidator,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    validate_simulation_database,
)


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult class."""
    
    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            check_name="test_check",
            severity=ValidationSeverity.ERROR,
            passed=False,
            message="Test failed",
            violation_count=5
        )
        
        self.assertEqual(result.check_name, "test_check")
        self.assertEqual(result.severity, ValidationSeverity.ERROR)
        self.assertFalse(result.passed)
        self.assertEqual(result.message, "Test failed")
        self.assertEqual(result.violation_count, 5)
    
    def test_validation_result_post_init(self):
        """Test that violation_count is set correctly in post_init."""
        result = ValidationResult(
            check_name="test_check",
            severity=ValidationSeverity.ERROR,
            passed=False,
            message="Test failed"
        )
        
        # Should default to 1 if not specified and not passed
        self.assertEqual(result.violation_count, 1)


class TestValidationReport(unittest.TestCase):
    """Test ValidationReport class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.report = ValidationReport(
            database_path="test.db",
            simulation_id="test_sim"
        )
    
    def test_report_creation(self):
        """Test creating a validation report."""
        self.assertEqual(self.report.database_path, "test.db")
        self.assertEqual(self.report.simulation_id, "test_sim")
        self.assertEqual(len(self.report.results), 0)
        self.assertIsNone(self.report.end_time)
    
    def test_add_result(self):
        """Test adding results to report."""
        result = ValidationResult(
            check_name="test_check",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="Test passed"
        )
        
        self.report.add_result(result)
        self.assertEqual(len(self.report.results), 1)
        self.assertEqual(self.report.results[0], result)
    
    def test_finalize(self):
        """Test finalizing a report."""
        self.assertIsNone(self.report.end_time)
        self.report.finalize()
        self.assertIsNotNone(self.report.end_time)
    
    def test_count_properties(self):
        """Test count properties."""
        # Add some test results
        self.report.add_result(ValidationResult(
            check_name="check1", severity=ValidationSeverity.INFO, passed=True, message="Passed"
        ))
        self.report.add_result(ValidationResult(
            check_name="check2", severity=ValidationSeverity.WARNING, passed=False, message="Warning"
        ))
        self.report.add_result(ValidationResult(
            check_name="check3", severity=ValidationSeverity.ERROR, passed=False, message="Error"
        ))
        
        self.assertEqual(self.report.total_checks, 3)
        self.assertEqual(self.report.passed_checks, 1)
        self.assertEqual(self.report.failed_checks, 2)
        self.assertEqual(self.report.error_count, 1)
        self.assertEqual(self.report.warning_count, 1)
        self.assertEqual(self.report.info_count, 0)
    
    def test_has_errors_warnings(self):
        """Test has_errors and has_warnings methods."""
        # No errors or warnings initially
        self.assertFalse(self.report.has_errors())
        self.assertFalse(self.report.has_warnings())
        
        # Add warning
        self.report.add_result(ValidationResult(
            check_name="warning", severity=ValidationSeverity.WARNING, passed=False, message="Warning"
        ))
        self.assertFalse(self.report.has_errors())
        self.assertTrue(self.report.has_warnings())
        
        # Add error
        self.report.add_result(ValidationResult(
            check_name="error", severity=ValidationSeverity.ERROR, passed=False, message="Error"
        ))
        self.assertTrue(self.report.has_errors())
        self.assertTrue(self.report.has_warnings())
    
    def test_is_clean(self):
        """Test is_clean method."""
        # Clean initially
        self.assertTrue(self.report.is_clean())
        
        # Add failed check
        self.report.add_result(ValidationResult(
            check_name="failed", severity=ValidationSeverity.ERROR, passed=False, message="Failed"
        ))
        self.assertFalse(self.report.is_clean())
    
    def test_get_summary(self):
        """Test get_summary method."""
        self.report.add_result(ValidationResult(
            check_name="test", severity=ValidationSeverity.INFO, passed=True, message="Passed"
        ))
        
        summary = self.report.get_summary()
        self.assertIn("Database Validation Summary", summary)
        self.assertIn("test.db", summary)
        self.assertIn("test_sim", summary)
        self.assertIn("Total Checks: 1", summary)
        self.assertIn("Passed: 1", summary)
    
    def test_to_dict(self):
        """Test to_dict method."""
        self.report.add_result(ValidationResult(
            check_name="test", severity=ValidationSeverity.INFO, passed=True, message="Passed"
        ))
        
        report_dict = self.report.to_dict()
        self.assertEqual(report_dict["database_path"], "test.db")
        self.assertEqual(report_dict["simulation_id"], "test_sim")
        self.assertIn("summary", report_dict)
        self.assertIn("results", report_dict)
        self.assertEqual(len(report_dict["results"]), 1)


class TestIntegrityValidator(unittest.TestCase):
    """Test IntegrityValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create in-memory database
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.report = ValidationReport()
        self.validator = IntegrityValidator(self.session, self.report)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.session.close()
        self.engine.dispose()
    
    def test_validate_table_existence(self):
        """Test table existence validation."""
        self.validator.validate_table_existence()
        
        # Should pass since we created all tables
        self.assertEqual(len(self.report.results), 1)
        result = self.report.results[0]
        self.assertEqual(result.check_name, "table_existence")
        self.assertTrue(result.passed)
    
    def test_validate_primary_keys(self):
        """Test primary key validation."""
        # Add some test data
        from farm.database.models import AgentModel, AgentStateModel
        
        agent = AgentModel(
            agent_id="test_agent",
            birth_time=0,
            agent_type="test",
            position_x=0.0,
            position_y=0.0,
            initial_resources=100.0,
            starting_health=100.0,
        )
        self.session.add(agent)
        self.session.commit()
        
        state = AgentStateModel(
            id="test_agent-0",
            agent_id="test_agent",
            step_number=0,
            position_x=0.0,
            position_y=0.0,
            position_z=0.0,
            resource_level=100.0,
            current_health=100.0,
            starting_health=100.0,
            starvation_counter=0,
            is_defending=False,
            total_reward=0.0,
            age=0
        )
        self.session.add(state)
        self.session.commit()
        
        self.validator.validate_primary_keys()
        
        # Should have 2 results (agents and agent_states)
        self.assertEqual(len(self.report.results), 2)
        for result in self.report.results:
            self.assertTrue(result.passed)
    
    def test_validate_foreign_keys(self):
        """Test foreign key validation."""
        # Add test data with valid references
        from farm.database.models import AgentModel, AgentStateModel
        
        agent = AgentModel(
            agent_id="test_agent",
            birth_time=0,
            agent_type="test",
            position_x=0.0,
            position_y=0.0,
            initial_resources=100.0,
            starting_health=100.0,
        )
        self.session.add(agent)
        self.session.commit()
        
        state = AgentStateModel(
            id="test_agent-0",
            agent_id="test_agent",
            step_number=0,
            position_x=0.0,
            position_y=0.0,
            position_z=0.0,
            resource_level=100.0,
            current_health=100.0,
            starting_health=100.0,
            starvation_counter=0,
            is_defending=False,
            total_reward=0.0,
            age=0
        )
        self.session.add(state)
        self.session.commit()
        
        self.validator.validate_foreign_keys()
        
        # Should have 2 results (agent_states and actions)
        self.assertEqual(len(self.report.results), 2)
        for result in self.report.results:
            self.assertTrue(result.passed)
    
    def test_validate_not_null_constraints(self):
        """Test NOT NULL constraint validation."""
        self.validator.validate_not_null_constraints()
        
        # Should have 2 results (agents and agent_states)
        self.assertEqual(len(self.report.results), 2)
        for result in self.report.results:
            self.assertTrue(result.passed)
    
    def test_validate_temporal_consistency(self):
        """Test temporal consistency validation."""
        # Add test data with valid temporal relationships
        from farm.database.models import AgentModel
        
        agent = AgentModel(
            agent_id="test_agent",
            birth_time=0,
            death_time=10,
            agent_type="test",
            position_x=0.0,
            position_y=0.0,
            initial_resources=100.0,
            starting_health=100.0,
        )
        self.session.add(agent)
        self.session.commit()
        
        self.validator.validate_temporal_consistency()
        
        # Should have 2 results (temporal_consistency and step_ordering)
        self.assertEqual(len(self.report.results), 2)
        for result in self.report.results:
            self.assertTrue(result.passed)


class TestStatisticalValidator(unittest.TestCase):
    """Test StatisticalValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create in-memory database
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.report = ValidationReport()
        self.validator = StatisticalValidator(self.session, self.report)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.session.close()
        self.engine.dispose()
    
    def test_validate_health_ranges(self):
        """Test health range validation."""
        # Add test data with valid health values
        from farm.database.models import AgentStateModel
        
        state = AgentStateModel(
            id="test_agent-0",
            agent_id="test_agent",
            step_number=0,
            position_x=0.0,
            position_y=0.0,
            position_z=0.0,
            resource_level=100.0,
            current_health=50.0,  # Valid: 0 <= 50 <= 100
            starting_health=100.0,
            starvation_counter=0,
            is_defending=False,
            total_reward=0.0,
            age=0
        )
        self.session.add(state)
        self.session.commit()
        
        self.validator.validate_health_ranges()
        
        self.assertEqual(len(self.report.results), 1)
        result = self.report.results[0]
        self.assertTrue(result.passed)
    
    def test_validate_resource_ranges(self):
        """Test resource range validation."""
        # Add test data with valid resource values
        from farm.database.models import AgentStateModel
        
        state = AgentStateModel(
            id="test_agent-0",
            agent_id="test_agent",
            step_number=0,
            position_x=0.0,
            position_y=0.0,
            position_z=0.0,
            resource_level=50.0,  # Valid: >= 0
            current_health=100.0,
            starting_health=100.0,
            starvation_counter=0,
            is_defending=False,
            total_reward=0.0,
            age=0
        )
        self.session.add(state)
        self.session.commit()
        
        self.validator.validate_resource_ranges()
        
        # Should have 2 results (agent_states and resource_states)
        self.assertEqual(len(self.report.results), 2)
        for result in self.report.results:
            self.assertTrue(result.passed)
    
    def test_validate_generation_monotonicity(self):
        """Test generation monotonicity validation."""
        # Add test data with valid generation progression
        from farm.database.models import AgentModel, ReproductionEventModel
        
        parent = AgentModel(
            agent_id="parent",
            birth_time=0,
            agent_type="test",
            position_x=0.0,
            position_y=0.0,
            initial_resources=100.0,
            starting_health=100.0,
            generation=1
        )
        self.session.add(parent)
        
        offspring = AgentModel(
            agent_id="offspring",
            birth_time=10,
            agent_type="test",
            position_x=0.0,
            position_y=0.0,
            initial_resources=50.0,
            starting_health=100.0,
            generation=2  # Valid: parent + 1
        )
        self.session.add(offspring)
        
        reproduction = ReproductionEventModel(
            step_number=10,
            parent_id="parent",
            offspring_id="offspring",
            success=True,
            parent_resources_before=100.0,
            parent_resources_after=50.0,
            offspring_initial_resources=50.0,
            parent_generation=1,
            offspring_generation=2,
            parent_position_x=0.0,
            parent_position_y=0.0
        )
        self.session.add(reproduction)
        self.session.commit()
        
        self.validator.validate_generation_monotonicity()
        
        self.assertEqual(len(self.report.results), 1)
        result = self.report.results[0]
        self.assertTrue(result.passed)
    
    def test_validate_population_consistency(self):
        """Test population consistency validation."""
        # Add test data
        from farm.database.models import AgentModel, SimulationStepModel, Simulation
        
        # Create simulation record first
        simulation = Simulation(
            simulation_id="test_sim",
            parameters={"test": "data"},
            simulation_db_path="test.db"
        )
        self.session.add(simulation)
        
        agent = AgentModel(
            agent_id="test_agent",
            simulation_id="test_sim",
            birth_time=0,
            agent_type="system",
            position_x=0.0,
            position_y=0.0,
            initial_resources=100.0,
            starting_health=100.0,
        )
        self.session.add(agent)
        
        step = SimulationStepModel(
            step_number=0,
            simulation_id="test_sim",
            total_agents=1,
            system_agents=1,
            independent_agents=0,
            control_agents=0,
            total_resources=100.0,
            average_agent_resources=100.0,
            births=0,
            deaths=0,
            current_max_generation=0,
            resource_efficiency=1.0,
            resource_distribution_entropy=0.0,
            average_agent_health=100.0,
            average_agent_age=0,
            average_reward=0.0,
            combat_encounters=0,
            successful_attacks=0,
            resources_shared=0.0,
            resources_shared_this_step=0.0,
            combat_encounters_this_step=0,
            successful_attacks_this_step=0,
            genetic_diversity=0.0,
            dominant_genome_ratio=1.0,
            resources_consumed=0.0
        )
        self.session.add(step)
        self.session.commit()
        
        self.validator.validate_population_consistency()
        
        self.assertEqual(len(self.report.results), 1)
        result = self.report.results[0]
        self.assertTrue(result.passed)
    
    def test_validate_step_continuity(self):
        """Test step continuity validation."""
        # Add test data with continuous steps
        from farm.database.models import SimulationStepModel, Simulation
        
        # Create simulation record first
        simulation = Simulation(
            simulation_id="test_sim",
            parameters={"test": "data"},
            simulation_db_path="test.db"
        )
        self.session.add(simulation)
        
        for i in range(3):
            step = SimulationStepModel(
                step_number=i,
                simulation_id="test_sim",
                total_agents=1,
                system_agents=1,
                independent_agents=0,
                control_agents=0,
                total_resources=100.0,
                average_agent_resources=100.0,
                births=0,
                deaths=0,
                current_max_generation=0,
                resource_efficiency=1.0,
                resource_distribution_entropy=0.0,
                average_agent_health=100.0,
                average_agent_age=0,
                average_reward=0.0,
                combat_encounters=0,
                successful_attacks=0,
                resources_shared=0.0,
                resources_shared_this_step=0.0,
                combat_encounters_this_step=0,
                successful_attacks_this_step=0,
                genetic_diversity=0.0,
                dominant_genome_ratio=1.0,
                resources_consumed=0.0
            )
            self.session.add(step)
        self.session.commit()
        
        self.validator.validate_step_continuity()
        
        self.assertEqual(len(self.report.results), 1)
        result = self.report.results[0]
        self.assertTrue(result.passed)


class TestValidateSimulationDatabase(unittest.TestCase):
    """Test main validation function."""
    
    def test_validate_nonexistent_database(self):
        """Test validation of non-existent database."""
        report = validate_simulation_database("/nonexistent/database.db")
        
        self.assertFalse(report.is_clean())
        self.assertTrue(report.has_errors())
        self.assertEqual(len(report.results), 1)
        self.assertEqual(report.results[0].check_name, "database_connection")
    
    def test_validate_empty_database(self):
        """Test validation of empty database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create empty database
            engine = create_engine(f"sqlite:///{tmp_path}")
            engine.dispose()
            
            report = validate_simulation_database(tmp_path)
            
            # Should have some failures due to missing tables
            self.assertFalse(report.is_clean())
            self.assertTrue(report.has_errors())
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_validate_valid_database(self):
        """Test validation of valid database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create database with schema
            engine = create_engine(f"sqlite:///{tmp_path}")
            Base.metadata.create_all(engine)
            engine.dispose()
            
            report = validate_simulation_database(tmp_path)
            
            # Should pass basic integrity checks
            self.assertTrue(report.is_clean())
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_validate_with_simulation_id(self):
        """Test validation with specific simulation ID."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create database with schema
            engine = create_engine(f"sqlite:///{tmp_path}")
            Base.metadata.create_all(engine)
            engine.dispose()
            
            report = validate_simulation_database(tmp_path, simulation_id="test_sim")
            
            self.assertEqual(report.simulation_id, "test_sim")
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_validate_selective_checks(self):
        """Test validation with selective checks."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create database with schema
            engine = create_engine(f"sqlite:///{tmp_path}")
            Base.metadata.create_all(engine)
            engine.dispose()
            
            # Only run integrity checks
            report = validate_simulation_database(
                tmp_path, 
                include_integrity=True, 
                include_statistical=False
            )
            
            # Should have fewer results than full validation
            self.assertGreater(len(report.results), 0)
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCLITool(unittest.TestCase):
    """Test CLI tool functionality."""
    
    def test_cli_help(self):
        """Test CLI help output."""
        import subprocess
        import sys
        from pathlib import Path
        
        # Use absolute path to avoid working directory issues
        script_path = Path(__file__).parent.parent.parent / "scripts" / "validate_simulation_db.py"
        
        result = subprocess.run([
            sys.executable, 
            str(script_path),
            "--help"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Validate simulation database", result.stdout)
    
    def test_cli_nonexistent_file(self):
        """Test CLI with non-existent file."""
        import subprocess
        import sys
        from pathlib import Path
        
        # Use absolute path to avoid working directory issues
        script_path = Path(__file__).parent.parent.parent / "scripts" / "validate_simulation_db.py"
        
        result = subprocess.run([
            sys.executable,
            str(script_path),
            "/nonexistent/database.db"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 3)  # Error exit code
        self.assertIn("does not exist", result.stderr)
    
    def test_cli_valid_database(self):
        """Test CLI with valid database."""
        import subprocess
        import sys
        from pathlib import Path
        
        # Use absolute path to avoid working directory issues
        script_path = Path(__file__).parent.parent.parent / "scripts" / "validate_simulation_db.py"
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create database with schema
            engine = create_engine(f"sqlite:///{tmp_path}")
            Base.metadata.create_all(engine)
            engine.dispose()
            
            result = subprocess.run([
                sys.executable,
                str(script_path),
                tmp_path
            ], capture_output=True, text=True)
            
            self.assertEqual(result.returncode, 0)  # Success exit code
            self.assertIn("All validations passed", result.stdout)
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_cli_json_output(self):
        """Test CLI with JSON output."""
        import subprocess
        import sys
        from pathlib import Path
        
        # Use absolute path to avoid working directory issues
        script_path = Path(__file__).parent.parent.parent / "scripts" / "validate_simulation_db.py"
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create database with schema
            engine = create_engine(f"sqlite:///{tmp_path}")
            Base.metadata.create_all(engine)
            engine.dispose()
            
            result = subprocess.run([
                sys.executable,
                str(script_path),
                "--json",
                tmp_path
            ], capture_output=True, text=True)
            
            self.assertEqual(result.returncode, 0)
            
            # Should be valid JSON
            output = json.loads(result.stdout)
            self.assertIn("summary", output)
            self.assertIn("results", output)
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
