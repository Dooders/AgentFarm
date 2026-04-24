"""Phylogenetics data processor.

Provides the data-processor entry point used by
:class:`~farm.analysis.phylogenetics.module.PhylogeneticsModule` to convert
raw simulation data into a
:class:`~farm.analysis.phylogenetics.compute.PhylogeneticTree`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from farm.database.models import AgentModel
from farm.analysis.common.utils import find_database_path
from farm.analysis.phylogenetics.compute import (
    PhylogeneticTree,
    build_phylogenetic_tree,
    build_phylogenetic_tree_from_records,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)

# Framework kwargs injected by BaseAnalysisModule.run_analysis() that must not
# be forwarded to the builder functions.
_FRAMEWORK_KWARGS = frozenset({"save_to_db", "db_path"})


def process_phylogenetics_data(data: Any, **kwargs: Any) -> PhylogeneticTree:
    """Convert raw data into a :class:`~farm.analysis.phylogenetics.compute.PhylogeneticTree`.

    Dispatch rules:

    * If ``data`` is a :class:`pathlib.Path` or :class:`str`, the function
      locates ``simulation.db`` inside the experiment directory, queries all
      :class:`~farm.database.models.AgentModel` rows, and calls
      :func:`~farm.analysis.phylogenetics.compute.build_phylogenetic_tree`.
    * If ``data`` is a SQLAlchemy :class:`~sqlalchemy.orm.Session`, query
      all :class:`~farm.database.models.AgentModel` rows and call
      :func:`~farm.analysis.phylogenetics.compute.build_phylogenetic_tree`.
    * If ``data`` is a list of dicts (e.g. ``evolution_lineage.json``
      records), call
      :func:`~farm.analysis.phylogenetics.compute.build_phylogenetic_tree_from_records`.
    * If ``data`` is already a :class:`~farm.analysis.phylogenetics.compute.PhylogeneticTree`,
      return it unchanged.
    * Otherwise return an empty tree.

    Parameters
    ----------
    data:
        Raw data source.
    **kwargs:
        Forwarded to the underlying builder (e.g. ``max_depth``).
        Framework-injected keys (``save_to_db``, ``db_path``) are silently
        consumed and not passed to the builders.

    Returns
    -------
    PhylogeneticTree
    """
    # Strip framework-injected kwargs before forwarding to builders
    builder_kwargs = {k: v for k, v in kwargs.items() if k not in _FRAMEWORK_KWARGS}

    if isinstance(data, PhylogeneticTree):
        return data

    if isinstance(data, (Path, str)):
        experiment_path = Path(data)
        try:
            db_path = find_database_path(experiment_path, "simulation.db")
        except FileNotFoundError:
            logger.warning(
                "process_phylogenetics_data: simulation.db not found in %s; returning empty tree",
                experiment_path,
            )
            return PhylogeneticTree(nodes={}, roots=[], is_dag=False)

        logger.info(
            "process_phylogenetics_data: loading agent lineage from database %s", db_path
        )
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            with Session(engine) as session:
                agents = session.query(AgentModel).all()
                return build_phylogenetic_tree(agents, **builder_kwargs)
        except Exception as exc:
            logger.warning("process_phylogenetics_data: DB query failed: %s", exc)
            return PhylogeneticTree(nodes={}, roots=[], is_dag=False)
        finally:
            engine.dispose()

    if isinstance(data, list):
        if not data:
            return PhylogeneticTree(nodes={}, roots=[], is_dag=False)

        first_item = next((item for item in data if item is not None), None)
        if first_item is None:
            return PhylogeneticTree(nodes={}, roots=[], is_dag=False)

        if isinstance(first_item, dict):
            return build_phylogenetic_tree_from_records(data, **builder_kwargs)

        if hasattr(first_item, "agent_id"):
            return build_phylogenetic_tree(data, **builder_kwargs)

        logger.warning(
            "process_phylogenetics_data: unsupported list item type %s; returning empty tree",
            type(first_item).__name__,
        )
        return PhylogeneticTree(nodes={}, roots=[], is_dag=False)

    # SQLAlchemy session duck-typed check
    if hasattr(data, "query"):
        try:
            agents = data.query(AgentModel).all()
            return build_phylogenetic_tree(agents, **builder_kwargs)
        except Exception as exc:
            logger.warning("process_phylogenetics_data: DB query failed: %s", exc)
            return PhylogeneticTree(nodes={}, roots=[], is_dag=False)

    logger.warning(
        "process_phylogenetics_data: unrecognised data type %s; returning empty tree",
        type(data).__name__,
    )
    return PhylogeneticTree(nodes={}, roots=[], is_dag=False)
