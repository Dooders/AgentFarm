"""Tests for ResearchDBClient and research_models module.

Uses SQLite temp files so no filesystem state is required.
The ResearchDBClient returns detached SQLAlchemy instances from its
context-manager sessions, so tests access the DB directly for IDs.
"""

import shutil
import tempfile
import unittest
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.database.models import Base
from farm.database.research_db_client import ResearchDBClient
from farm.database.research_models import ExperimentStats, IterationStats, Research


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pop_stats(**overrides):
    base = {"mean": 50.0, "std": 5.0, "max": 80.0, "min": 20.0}
    base.update(overrides)
    return base


def _res_stats(**overrides):
    base = {
        "mean_resources": 100.0,
        "std_resources": 10.0,
        "mean_efficiency": 0.8,
        "std_efficiency": 0.05,
    }
    base.update(overrides)
    return base


def _repro_stats(**overrides):
    base = {
        "mean_success_rate": 0.5,
        "std_success_rate": 0.1,
        "total_attempts": 200,
        "total_successes": 100,
    }
    base.update(overrides)
    return base


def _get_research_id(client, name: str) -> int:
    """Retrieve the integer id of a Research record by name using a fresh session."""
    with client.Session() as session:
        r = session.query(Research).filter_by(name=name).first()
        return r.id if r else None


def _get_experiment_stats_id(client, experiment_id: str) -> int:
    """Retrieve integer id of ExperimentStats by experiment_id string."""
    with client.Session() as session:
        e = session.query(ExperimentStats).filter_by(experiment_id=experiment_id).first()
        return e.id if e else None


# ---------------------------------------------------------------------------
# ResearchDBClient model / ORM tests
# ---------------------------------------------------------------------------


class TestResearchModels(unittest.TestCase):
    """Tests for ORM model definitions."""

    def setUp(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def tearDown(self):
        self.session.close()

    def test_research_model_columns(self):
        r = Research(name="test_project", description="desc", parameters={})
        self.session.add(r)
        self.session.commit()
        fetched = self.session.query(Research).filter_by(name="test_project").first()
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "test_project")

    def test_experiment_stats_model(self):
        r = Research(name="proj", parameters={})
        self.session.add(r)
        self.session.flush()

        es = ExperimentStats(
            research_id=r.id,
            experiment_id="exp_001",
            timestamp=datetime.now(),
            num_iterations=10,
            mean_population=50.0,
            std_population=5.0,
            max_population=80.0,
            min_population=20.0,
            mean_resources=100.0,
            std_resources=10.0,
            mean_efficiency=0.8,
            std_efficiency=0.05,
            mean_success_rate=0.5,
            std_success_rate=0.1,
            total_reproduction_attempts=200,
            total_successful_reproductions=100,
        )
        self.session.add(es)
        self.session.commit()
        fetched = self.session.query(ExperimentStats).filter_by(
            experiment_id="exp_001"
        ).first()
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.mean_population, 50.0)

    def test_iteration_stats_model(self):
        r = Research(name="proj2", parameters={})
        self.session.add(r)
        self.session.flush()

        es = ExperimentStats(
            research_id=r.id,
            experiment_id="exp_002",
            timestamp=datetime.now(),
            num_iterations=5,
            mean_population=40.0,
            std_population=4.0,
            max_population=60.0,
            min_population=10.0,
            mean_resources=80.0,
            std_resources=8.0,
            mean_efficiency=0.75,
            std_efficiency=0.04,
            mean_success_rate=0.4,
            std_success_rate=0.09,
            total_reproduction_attempts=100,
            total_successful_reproductions=40,
        )
        self.session.add(es)
        self.session.flush()

        it = IterationStats(
            experiment_id=es.id,
            iteration_id="iter_001",
            avg_population=42.0,
            max_population=55,
            min_population=12,
            avg_resources=85.0,
            resource_efficiency=0.76,
            reproduction_attempts=20,
            successful_reproductions=8,
            reproduction_rate=0.4,
        )
        self.session.add(it)
        self.session.commit()
        fetched = self.session.query(IterationStats).filter_by(
            iteration_id="iter_001"
        ).first()
        self.assertIsNotNone(fetched)
        self.assertAlmostEqual(fetched.avg_population, 42.0)


# ---------------------------------------------------------------------------
# ResearchDBClient – get_or_create_research
# ---------------------------------------------------------------------------


class TestResearchDBClientGetOrCreate(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import os
        self.client = ResearchDBClient(os.path.join(self._tmpdir, "test.db"))

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_creates_new_research(self):
        self.client.get_or_create_research("my_project", "desc")
        rid = _get_research_id(self.client, "my_project")
        self.assertIsNotNone(rid)

    def test_returns_existing_research(self):
        self.client.get_or_create_research("duplicate")
        self.client.get_or_create_research("duplicate")
        # Only one record should exist
        with self.client.Session() as session:
            count = session.query(Research).filter_by(name="duplicate").count()
        self.assertEqual(count, 1)

    def test_creates_with_parameters(self):
        params = {"learning_rate": 0.01, "iterations": 100}
        self.client.get_or_create_research("params_proj", parameters=params)
        rid = _get_research_id(self.client, "params_proj")
        self.assertIsNotNone(rid)


# ---------------------------------------------------------------------------
# ResearchDBClient – add_experiment_stats / add_iteration_stats
# ---------------------------------------------------------------------------


class TestResearchDBClientAddStats(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import os
        self.client = ResearchDBClient(os.path.join(self._tmpdir, "test.db"))
        self.client.get_or_create_research("stats_project")
        self.research_id = _get_research_id(self.client, "stats_project")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_add_experiment_stats(self):
        self.client.add_experiment_stats(
            research_id=self.research_id,
            experiment_id="exp_001",
            num_iterations=10,
            population_stats=_pop_stats(),
            resource_stats=_res_stats(),
            reproduction_stats=_repro_stats(),
            description="baseline experiment",
        )
        eid = _get_experiment_stats_id(self.client, "exp_001")
        self.assertIsNotNone(eid)

    def test_add_iteration_stats(self):
        self.client.add_experiment_stats(
            research_id=self.research_id,
            experiment_id="exp_002",
            num_iterations=3,
            population_stats=_pop_stats(),
            resource_stats=_res_stats(),
            reproduction_stats=_repro_stats(),
        )
        eid = _get_experiment_stats_id(self.client, "exp_002")

        self.client.add_iteration_stats(
            experiment_id=eid,
            iteration_id="iter_001",
            population_stats={"avg": 48.0, "max": 75, "min": 22},
            resource_stats={"avg_resources": 95.0, "efficiency": 0.78},
            reproduction_stats={"attempts": 50, "successes": 25, "success_rate": 0.5},
        )
        with self.client.Session() as session:
            it = session.query(IterationStats).filter_by(iteration_id="iter_001").first()
            self.assertIsNotNone(it)
            self.assertEqual(it.iteration_id, "iter_001")


# ---------------------------------------------------------------------------
# ResearchDBClient – get_experiment_stats / get_iteration_stats
# ---------------------------------------------------------------------------


class TestResearchDBClientGetStats(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import os
        self.client = ResearchDBClient(os.path.join(self._tmpdir, "test.db"))
        self.client.get_or_create_research("get_stats_project")
        self.research_id = _get_research_id(self.client, "get_stats_project")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_get_experiment_stats_empty(self):
        results = self.client.get_experiment_stats(self.research_id)
        self.assertEqual(results, [])

    def test_get_experiment_stats_returns_all(self):
        for i in range(3):
            self.client.add_experiment_stats(
                research_id=self.research_id,
                experiment_id=f"exp_{i}",
                num_iterations=5,
                population_stats=_pop_stats(),
                resource_stats=_res_stats(),
                reproduction_stats=_repro_stats(),
            )
        results = self.client.get_experiment_stats(self.research_id)
        self.assertEqual(len(results), 3)

    def test_get_iteration_stats_empty(self):
        self.client.add_experiment_stats(
            research_id=self.research_id,
            experiment_id="exp_empty",
            num_iterations=1,
            population_stats=_pop_stats(),
            resource_stats=_res_stats(),
            reproduction_stats=_repro_stats(),
        )
        eid = _get_experiment_stats_id(self.client, "exp_empty")
        iters = self.client.get_iteration_stats(eid)
        self.assertEqual(iters, [])

    def test_get_iteration_stats_returns_all(self):
        self.client.add_experiment_stats(
            research_id=self.research_id,
            experiment_id="exp_with_iters",
            num_iterations=2,
            population_stats=_pop_stats(),
            resource_stats=_res_stats(),
            reproduction_stats=_repro_stats(),
        )
        eid = _get_experiment_stats_id(self.client, "exp_with_iters")
        for j in range(2):
            self.client.add_iteration_stats(
                experiment_id=eid,
                iteration_id=f"iter_{j}",
                population_stats={"avg": 45.0, "max": 70, "min": 15},
                resource_stats={"avg_resources": 90.0, "efficiency": 0.7},
                reproduction_stats={"attempts": 30, "successes": 15, "success_rate": 0.5},
            )
        iters = self.client.get_iteration_stats(eid)
        self.assertEqual(len(iters), 2)


if __name__ == "__main__":
    unittest.main()
