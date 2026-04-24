"""Phylogenetics analysis module registration.

Registers the phylogenetics analysis module with the analysis framework so it
is discoverable via :class:`~farm.analysis.service.AnalysisService` and the
module registry.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.phylogenetics.data import process_phylogenetics_data
from farm.analysis.phylogenetics.analyze import analyze_phylogenetics
from farm.analysis.phylogenetics.plot import plot_phylogenetic_tree


class PhylogeneticsModule(BaseAnalysisModule):
    """Analysis module for phylogenetic tree construction and visualisation.

    Supports two data sources:

    * **Simulation database** – queries all agents from the DB and builds a
      phylogenetic tree from their ``genome_id`` lineage.
    * **Evolution-experiment result** – accepts the list of records from
      ``evolution_lineage.json`` and builds the tree from ``parent_ids`` fields.
    """

    def __init__(self) -> None:
        super().__init__(
            name="phylogenetics",
            description=(
                "Phylogenetic tree/DAG construction, serialisation, and visualisation "
                "from GenomeId lineage (simulation DB or evolution_lineage.json)"
            ),
        )

        validator = CompositeValidator(
            [
                ColumnValidator(required_columns=[], column_types={}),
                DataQualityValidator(min_rows=0),
            ]
        )
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all phylogenetics analysis functions."""
        self._functions = {
            "analyze_phylogenetics": make_analysis_function(analyze_phylogenetics),
            "plot_phylogenetic_tree": make_analysis_function(plot_phylogenetic_tree),
        }

        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [self._functions["analyze_phylogenetics"]],
            "plots": [self._functions["plot_phylogenetic_tree"]],
            "basic": [
                self._functions["analyze_phylogenetics"],
                self._functions["plot_phylogenetic_tree"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Return the data processor for phylogenetics analysis."""
        return SimpleDataProcessor(process_phylogenetics_data)

    def supports_database(self) -> bool:
        """This module can use a simulation database."""
        return True

    def get_db_filename(self) -> str:
        """Database filename used by this module."""
        return "simulation.db"


# Singleton instance consumed by the module registry
phylogenetics_module = PhylogeneticsModule()
