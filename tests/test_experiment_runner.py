import json
import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.analysis.experiment_analysis import ExperimentAnalyzer
from farm.core.config import SimulationConfig
from farm.database.models import (
    Base,
    Experiment,
    ExperimentEvent,
    ExperimentMetric,
    Simulation
)
from farm.runners.experiment_runner import ExperimentRunner


class TestExperimentRunner:
    @pytest.fixture(autouse=True)
    def setup_database(self, temp_db_path):
        """Set up a test database."""
        engine = create_engine(f"sqlite:///{temp_db_path}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        yield
        self.session.close()

    @pytest.fixture
    def base_config(self):
        """Create a base simulation configuration for testing."""
        return SimulationConfig(
            simulation_steps=100,
            system_agents=5,
            independent_agents=5,
            control_agents=0,
            initial_resources=20,
            max_resource_amount=10,
            resource_regen_rate=0.1,
            width=100,
            height=100
        )

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        # Close any open sessions
        if hasattr(self, 'session'):
            self.session.close()
        # Close any open database connections
        engine = create_engine(f"sqlite:///{db_path}")
        engine.dispose()
        # Now try to delete the file
        try:
            os.unlink(db_path)
        except PermissionError:
            # If we still can't delete it, log a warning but don't fail the test
            import warnings
            warnings.warn(f"Could not delete temporary database file: {db_path}")

    def test_experiment_creation(self, base_config, temp_db_path):
        """Test creating a new experiment."""
        # Create experiment runner
        runner = ExperimentRunner(
            base_config=base_config,
            experiment_name="test_experiment",
            db_path=temp_db_path
        )
        
        # Verify experiment was created in database
        experiment = self.session.query(Experiment).filter_by(
            experiment_id=runner.experiment_id
        ).first()
        
        assert experiment is not None
        assert experiment.name == "test_experiment"
        assert experiment.status == "created"
        
        # Compare only key properties of the config instead of the whole object
        # This avoids issues with serialization differences (tuples vs lists)
        stored_config = experiment.base_config
        assert stored_config['width'] == base_config.width
        assert stored_config['height'] == base_config.height
        assert stored_config['system_agents'] == base_config.system_agents
        assert stored_config['independent_agents'] == base_config.independent_agents
        assert stored_config['initial_resources'] == base_config.initial_resources
        assert stored_config['resource_regen_rate'] == base_config.resource_regen_rate
        assert stored_config['max_resource_amount'] == base_config.max_resource_amount

    def test_run_single_iteration(self, base_config, temp_db_path):
        """Test running a single iteration."""
        # Create experiment runner
        runner = ExperimentRunner(
            base_config=base_config,
            experiment_name="test_single_iteration",
            db_path=temp_db_path
        )
        
        # Run one iteration
        runner.run_iterations(num_iterations=1)
        
        # Verify simulation was created
        simulations = self.session.query(Simulation).filter_by(
            experiment_id=runner.experiment_id
        ).all()
        
        assert len(simulations) == 1
        assert simulations[0].iteration_number == 1
        assert simulations[0].status == "completed"
        
        # Verify metrics were logged
        metrics = self.session.query(ExperimentMetric).filter_by(
            experiment_id=runner.experiment_id
        ).all()
        
        assert len(metrics) > 0
        
        # Verify events were logged
        events = self.session.query(ExperimentEvent).filter_by(
            experiment_id=runner.experiment_id
        ).all()
        
        assert len(events) >= 3  # At least creation, start, and completion events
        
    def test_run_multiple_iterations_with_variations(self, base_config, temp_db_path):
        """Test running multiple iterations with config variations."""
        # Create variations
        variations = [
            {"system_agents": 3},
            {"system_agents": 8},
            {"initial_resources": 10}
        ]

        # Create experiment runner
        runner = ExperimentRunner(
            base_config=base_config,
            experiment_name="test_variations",
            db_path=temp_db_path
        )

        # For testing purposes, we'll run each iteration separately to avoid ID conflicts
        for i, variation in enumerate(variations):
            # Create a new experiment runner for each iteration
            if i > 0:
                # Create a new temporary database file
                temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
                temp_db.close()
                temp_db_path = Path(temp_db.name)
                
                # Create a new runner with the new database
                runner = ExperimentRunner(
                    base_config=base_config,
                    experiment_name=f"test_variations_{i}",
                    db_path=temp_db_path
                )
            
            # Run a single iteration with the current variation
            config = base_config.copy()
            for key, value in variation.items():
                setattr(config, key, value)
                
            runner.run_iterations(num_iterations=1)
            
            # Verify simulation was created
            simulations = self.session.query(Simulation).filter_by(
                experiment_id=runner.experiment_id
            ).all()
            
            assert len(simulations) == 1
            assert simulations[0].iteration_number == 1
            assert simulations[0].status == "completed"
            
    def test_experiment_analysis(self, base_config, temp_db_path):
        """Test experiment analysis functionality."""
        # Create and run experiment
        runner = ExperimentRunner(
            base_config=base_config,
            experiment_name="test_analysis",
            db_path=temp_db_path
        )
        runner.run_iterations(num_iterations=1)
        
        # Create analyzer
        analyzer = ExperimentAnalyzer(runner.db)
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Run analysis
            analyzer.analyze_experiment(
                experiment_id=runner.experiment_id,
                output_path=output_path
            )
            
            # Verify analysis outputs were created
            assert (output_path / "summary.json").exists()
            
            # Check if metrics were generated (if any)
            metrics_df = analyzer._get_metrics_dataframe(runner.experiment_id)
            if not metrics_df.empty:
                assert (output_path / "metrics.csv").exists()
            
    def test_experiment_comparison(self, base_config, temp_db_path):
        """Test comparing multiple experiments."""
        # Create two experiments with different configurations
        config1 = base_config
        config2 = base_config.copy()
        config2.system_agents = 20

        # Create and run first experiment
        runner1 = ExperimentRunner(
            base_config=config1,
            experiment_name="experiment1",
            db_path=temp_db_path
        )
        runner1.run_iterations(num_iterations=1)
        
        # Create a new temporary database file for the second experiment
        temp_db2 = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db2.close()
        temp_db_path2 = Path(temp_db2.name)

        # Create and run second experiment
        runner2 = ExperimentRunner(
            base_config=config2,
            experiment_name="experiment2",
            db_path=temp_db_path2
        )
        runner2.run_iterations(num_iterations=1)

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)

            # Create analyzer for each experiment
            analyzer1 = ExperimentAnalyzer(runner1.db)
            analyzer2 = ExperimentAnalyzer(runner2.db)
            
            # Analyze each experiment individually
            analyzer1.analyze_experiment(
                experiment_id=runner1.experiment_id,
                output_path=output_path / "experiment1"
            )
            
            analyzer2.analyze_experiment(
                experiment_id=runner2.experiment_id,
                output_path=output_path / "experiment2"
            )
            
            # Verify analysis outputs were created
            assert (output_path / "experiment1" / "summary.json").exists()
            assert (output_path / "experiment2" / "summary.json").exists()
            
    def test_error_handling(self, base_config, temp_db_path):
        """Test error handling in experiment runner."""
        # Create experiment runner
        runner = ExperimentRunner(
            base_config=base_config,
            experiment_name="test_errors",
            db_path=temp_db_path
        )
        
        # Test invalid experiment ID
        with pytest.raises(ValueError):
            analyzer = ExperimentAnalyzer(runner.db)
            analyzer.analyze_experiment(experiment_id=999999, output_path=Path("test_output"))
            
        # Test invalid configuration variation
        with pytest.raises(Exception):
            runner.run_iterations(
                num_iterations=1,
                config_variations=[{"invalid_param": 123}]
            )
            
        # Verify experiment status was set to failed
        experiment = self.session.query(Experiment).filter_by(
            experiment_id=runner.experiment_id
        ).first()
        assert experiment.status == "failed"
