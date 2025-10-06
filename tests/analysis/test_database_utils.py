"""
Comprehensive tests for database utilities module.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call

from farm.analysis.database_utils import (
    import_data_generic,
    import_multi_table_data,
    create_simulation_if_not_exists,
)


class TestImportDataGeneric:
    """Test the import_data_generic function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({"iteration": [1, 2, 3], "value": [10.0, 20.0, 30.0], "name": ["test1", "test2", "test3"]})

    def test_import_data_generic_success(self, sample_dataframe):
        """Test successful data import with iteration mapping."""
        mock_session = MagicMock()
        mock_model = MagicMock()
        mock_model.__name__ = "TestModel"

        # Mock iteration to ID mapping
        iteration_to_id = {1: 101, 2: 102, 3: 103}

        # Mock create function
        def create_func(row, sim_id):
            return mock_model(value=row["value"], simulation_id=sim_id)

        # Mock session query to return no existing records
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        count = import_data_generic(
            df=sample_dataframe,
            session=mock_session,
            model_class=mock_model,
            create_object_func=create_func,
            log_prefix="test data",
            iteration_to_id=iteration_to_id,
        )

        assert count == 3
        assert mock_session.add.call_count == 3
        mock_session.commit.assert_called()

    def test_import_data_generic_skip_existing(self, sample_dataframe):
        """Test skipping existing records."""
        mock_session = MagicMock()
        mock_model = MagicMock()
        mock_model.__name__ = "TestModel"

        iteration_to_id = {1: 101, 2: 102, 3: 103}

        def create_func(row, sim_id):
            return mock_model(value=row["value"], simulation_id=sim_id)

        # Mock existing record for first row
        mock_existing = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            mock_existing,  # First call returns existing
            None,  # Second call returns None
            None,  # Third call returns None
        ]

        count = import_data_generic(
            df=sample_dataframe,
            session=mock_session,
            model_class=mock_model,
            create_object_func=create_func,
            log_prefix="test data",
            iteration_to_id=iteration_to_id,
        )

        assert count == 2  # Only 2 new records added
        assert mock_session.add.call_count == 2

    def test_import_data_generic_missing_simulation_id(self, sample_dataframe):
        """Test handling missing simulation ID."""
        mock_session = MagicMock()
        mock_model = MagicMock()

        # Missing ID for iteration 2
        iteration_to_id = {1: 101, 3: 103}

        def create_func(row, sim_id):
            return mock_model(value=row["value"], simulation_id=sim_id)

        # Mock no existing records
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        count = import_data_generic(
            df=sample_dataframe,
            session=mock_session,
            model_class=mock_model,
            create_object_func=create_func,
            log_prefix="test data",
            iteration_to_id=iteration_to_id,
        )

        assert count == 2  # Only iterations 1 and 3 processed
        assert mock_session.add.call_count == 2

    def test_import_data_generic_batch_commits(self, sample_dataframe):
        """Test batch commits every 100 records."""
        mock_session = MagicMock()
        mock_model = MagicMock()

        # Create larger dataset
        large_df = pd.DataFrame({"iteration": range(150), "value": range(150)})

        iteration_to_id = {i: i + 100 for i in range(150)}

        def create_func(row, sim_id):
            return mock_model(value=row["value"], simulation_id=sim_id)

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        count = import_data_generic(
            df=large_df,
            session=mock_session,
            model_class=mock_model,
            create_object_func=create_func,
            log_prefix="test data",
            iteration_to_id=iteration_to_id,
            commit_batch_size=50,
        )

        assert count == 150
        # Should commit 4 times (at 50, 100, 150, and final)
        assert mock_session.commit.call_count == 4

    def test_import_data_generic_no_iteration_mapping(self, sample_dataframe):
        """Test import without iteration mapping."""
        mock_session = MagicMock()
        mock_model = MagicMock()

        def create_func(row, sim_id):
            assert sim_id is None  # Should be None when no mapping provided
            return mock_model(value=row["value"])

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        count = import_data_generic(
            df=sample_dataframe,
            session=mock_session,
            model_class=mock_model,
            create_object_func=create_func,
            log_prefix="test data",
        )

        assert count == 3
        assert mock_session.add.call_count == 3

    def test_import_data_generic_sqlalchemy_error(self, sample_dataframe):
        """Test handling SQLAlchemy errors."""
        mock_session = MagicMock()
        mock_model = MagicMock()

        iteration_to_id = {1: 101}

        def create_func(row, sim_id):
            return mock_model(value=row["value"], simulation_id=sim_id)

        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_session.commit.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            import_data_generic(
                df=sample_dataframe,
                session=mock_session,
                model_class=mock_model,
                create_object_func=create_func,
                log_prefix="test data",
                iteration_to_id=iteration_to_id,
            )


class TestImportMultiTableData:
    """Test the import_multi_table_data function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for multi-table testing."""
        return pd.DataFrame({"iteration": [1, 2, 3], "metric1": [10.0, 20.0, 30.0], "metric2": [100, 200, 300]})

    def test_import_multi_table_success(self, sample_dataframe):
        """Test successful multi-table data import."""
        mock_session = MagicMock()
        mock_sim_model = MagicMock()
        mock_data_model1 = MagicMock()
        mock_data_model2 = MagicMock()

        # Mock simulation creation
        mock_sim_instance = MagicMock()
        mock_sim_instance.id = 101
        mock_sim_model.return_value = mock_sim_instance

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        data_model_configs = [
            {
                "model_class": mock_data_model1,
                "create_func": lambda row, sim_id: mock_data_model1(metric1=row["metric1"], simulation_id=sim_id),
                "name": "data model 1",
            },
            {
                "model_class": mock_data_model2,
                "create_func": lambda row, sim_id: mock_data_model2(metric2=row["metric2"], simulation_id=sim_id),
                "name": "data model 2",
            },
        ]

        count = import_multi_table_data(
            df=sample_dataframe,
            session=mock_session,
            simulation_model_class=mock_sim_model,
            data_model_configs=data_model_configs,
            log_prefix="multi-table test",
        )

        assert count == 3
        assert mock_sim_model.call_count == 3  # 3 simulations created
        assert mock_session.add.call_count == 9  # 3 sims + 3*2 data objects
        assert mock_session.commit.call_count == 1  # Final commit

    def test_import_multi_table_simulation_error(self, sample_dataframe):
        """Test handling simulation creation errors."""
        mock_session = MagicMock()
        mock_sim_model = MagicMock()

        # Mock simulation creation failure
        mock_session.flush.side_effect = Exception("Simulation creation failed")

        data_model_configs = [
            {"model_class": MagicMock(), "create_func": lambda row, sim_id: MagicMock(), "name": "test model"}
        ]

        with pytest.raises(Exception, match="Simulation creation failed"):
            import_multi_table_data(
                df=sample_dataframe,
                session=mock_session,
                simulation_model_class=mock_sim_model,
                data_model_configs=data_model_configs,
                log_prefix="test",
            )

    def test_import_multi_table_data_error(self, sample_dataframe):
        """Test handling data object creation errors."""
        mock_session = MagicMock()
        mock_sim_model = MagicMock()
        mock_sim_instance = MagicMock()
        mock_sim_instance.id = 101
        mock_sim_model.return_value = mock_sim_instance

        def failing_create_func(row, sim_id):
            raise ValueError("Data creation failed")

        data_model_configs = [{"model_class": MagicMock(), "create_func": failing_create_func, "name": "failing model"}]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Should rollback and not count the failed simulation
        count = import_multi_table_data(
            df=sample_dataframe.head(1),  # Just one row
            session=mock_session,
            simulation_model_class=mock_sim_model,
            data_model_configs=data_model_configs,
            log_prefix="test",
        )

        assert count == 0  # No successful imports
        mock_session.rollback.assert_called()

    def test_import_multi_table_batch_commits(self, sample_dataframe):
        """Test batch commits in multi-table import."""
        mock_session = MagicMock()
        mock_sim_model = MagicMock()
        mock_sim_instance = MagicMock()
        mock_sim_instance.id = 101
        mock_sim_model.return_value = mock_sim_instance

        # Create larger dataset
        large_df = pd.DataFrame({"iteration": range(150), "metric1": range(150)})

        data_model_configs = [
            {"model_class": MagicMock(), "create_func": lambda row, sim_id: MagicMock(), "name": "test model"}
        ]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        count = import_multi_table_data(
            df=large_df,
            session=mock_session,
            simulation_model_class=mock_sim_model,
            data_model_configs=data_model_configs,
            log_prefix="test",
            commit_batch_size=50,
        )

        assert count == 150
        # Should commit 4 times (at 50, 100, 150, and final)
        assert mock_session.commit.call_count == 4


class TestCreateSimulationIfNotExists:
    """Test the create_simulation_if_not_exists function."""

    def test_create_new_simulations(self):
        """Test creating new simulations."""
        mock_session = MagicMock()
        mock_sim_model = MagicMock()

        sample_df = pd.DataFrame({"iteration": [1, 2, 3]})

        # Mock no existing simulations
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Mock new simulation instances with different IDs
        mock_sim_model.side_effect = [
            MagicMock(id=101),  # First call returns sim with id=101
            MagicMock(id=102),  # Second call returns sim with id=102
            MagicMock(id=103),  # Third call returns sim with id=103
        ]

        result = create_simulation_if_not_exists(
            df=sample_df, session=mock_session, simulation_model_class=mock_sim_model, log_prefix="test sim"
        )

        assert result == {1: 101, 2: 102, 3: 103}
        assert mock_sim_model.call_count == 3
        assert mock_session.add.call_count == 3

    def test_skip_existing_simulations(self):
        """Test skipping existing simulations."""
        mock_session = MagicMock()
        mock_sim_model = MagicMock()

        sample_df = pd.DataFrame({"iteration": [1, 2, 3]})

        # Mock existing simulation for iteration 2
        existing_sim = MagicMock()
        existing_sim.id = 202

        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            None,  # No existing for iteration 1
            existing_sim,  # Existing for iteration 2
            None,  # No existing for iteration 3
        ]

        # Mock new simulation instances for iterations 1 and 3
        mock_sim_model.side_effect = [
            MagicMock(id=101),  # For iteration 1
            MagicMock(id=103),  # For iteration 3
        ]

        result = create_simulation_if_not_exists(
            df=sample_df, session=mock_session, simulation_model_class=mock_sim_model, log_prefix="test sim"
        )

        assert result == {1: 101, 2: 202, 3: 103}
        assert mock_sim_model.call_count == 2  # Only 2 new simulations created
        assert mock_session.add.call_count == 2

    def test_simulation_creation_error(self):
        """Test handling simulation creation errors."""
        mock_session = MagicMock()
        mock_sim_model = MagicMock()

        sample_df = pd.DataFrame({"iteration": [1]})

        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        # The function catches SQLAlchemyError, not generic Exception
        from sqlalchemy.exc import SQLAlchemyError

        mock_session.flush.side_effect = SQLAlchemyError("Creation failed")

        result = create_simulation_if_not_exists(
            df=sample_df, session=mock_session, simulation_model_class=mock_sim_model, log_prefix="test sim"
        )

        # Should return empty mapping when creation fails
        assert result == {}
        assert mock_session.rollback.called


class TestIntegrationWithRefactoredCode:
    """Test integration with the refactored import functions."""

    def test_advantage_import_integration(self):
        """Test that the advantage import functions still work after refactoring."""
        from farm.analysis.advantage.import_csv_to_db import import_resource_acquisition_data

        # Test that the function exists and has correct signature
        assert callable(import_resource_acquisition_data)

        # Test with mocked dependencies
        mock_session = MagicMock()
        mock_df = pd.DataFrame({"iteration": [1, 2], "system_early_phase_resources": [10.0, 20.0]})

        # Mock no existing records so import will proceed
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        with patch("farm.analysis.advantage.import_csv_to_db.import_data_generic") as mock_import:
            mock_import.return_value = 2

            # This should work without errors
            import_resource_acquisition_data(mock_df, {1: 101, 2: 102}, mock_session)

            mock_import.assert_called_once()

    def test_dominance_import_integration(self):
        """Test that the dominance import functions still work after refactoring."""
        from farm.analysis.dominance.analyze import save_dominance_data_to_db

        # Test that the function exists
        assert callable(save_dominance_data_to_db)

        # Test with mocked dependencies
        mock_df = pd.DataFrame({"iteration": [1, 2], "population_dominance": ["system", "independent"]})

        with patch("farm.analysis.dominance.analyze.import_multi_table_data") as mock_import:
            mock_import.return_value = 2

            with patch("farm.analysis.dominance.analyze.init_db") as mock_init:
                with patch("farm.analysis.dominance.analyze.get_session") as mock_get:
                    mock_session = MagicMock()
                    mock_get.return_value = mock_session

                    # This should work without errors
                    result = save_dominance_data_to_db(mock_df, "sqlite:///:memory:")

                    assert result is True
                    mock_import.assert_called_once()


class TestErrorHandling:
    """Test error handling in database utilities."""

    def test_import_generic_with_invalid_data(self):
        """Test import with invalid data that causes create function to fail."""
        mock_session = MagicMock()
        mock_model = MagicMock()

        df = pd.DataFrame({"iteration": [1], "invalid_column": [None]})

        iteration_to_id = {1: 101}

        def failing_create_func(row, sim_id):
            raise ValueError("Invalid data")

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Invalid data"):
            import_data_generic(
                df=df,
                session=mock_session,
                model_class=mock_model,
                create_object_func=failing_create_func,
                log_prefix="test",
                iteration_to_id=iteration_to_id,
            )

    def test_multi_table_with_empty_config(self):
        """Test multi-table import with empty configuration."""
        mock_session = MagicMock()
        mock_sim_model = MagicMock()

        df = pd.DataFrame({"iteration": [1]})

        count = import_multi_table_data(
            df=df, session=mock_session, simulation_model_class=mock_sim_model, data_model_configs=[], log_prefix="test"
        )

        assert count == 1  # Still creates the simulation
        assert mock_sim_model.called

    def test_simulation_creation_with_duplicate_iterations(self):
        """Test simulation creation with duplicate iterations."""
        mock_session = MagicMock()
        mock_sim_model = MagicMock()

        df = pd.DataFrame(
            {
                "iteration": [1, 1, 2]  # Duplicate iteration 1
            }
        )

        # First call returns None (new), second returns existing sim
        existing_sim = MagicMock()
        existing_sim.id = 101

        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            None,  # First iteration 1 - create new
            existing_sim,  # Second iteration 1 - already exists
            None,  # iteration 2 - create new
        ]

        # Mock simulation creation for the new iterations
        mock_sim_model.side_effect = [
            MagicMock(id=101),  # For first occurrence of iteration 1
            MagicMock(id=102),  # For iteration 2
        ]

        result = create_simulation_if_not_exists(
            df=df, session=mock_session, simulation_model_class=mock_sim_model, log_prefix="test"
        )

        # Should create only 2 unique simulations
        assert len(result) == 2
        assert result[1] == 101  # From existing (second occurrence skipped)
        assert result[2] == 102  # New one
