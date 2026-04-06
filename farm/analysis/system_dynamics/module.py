"""
System dynamics analysis module: merged population, resources, and temporal view.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.system_dynamics.data import process_system_dynamics_data
from farm.analysis.system_dynamics.analyze import (
    run_foundation_analyses,
    analyze_system_dynamics_synthesis,
    write_unified_system_dynamics_report,
)


class SystemDynamicsModule(BaseAnalysisModule):
    """Cross-domain synthesis over population, resource, and temporal signals."""

    def __init__(self):
        super().__init__(
            name="system_dynamics",
            description=(
                "System dynamics view: merged population, resource, and temporal series; "
                "cross-domain correlations, Granger tests, and feedback-loop heuristics"
            ),
        )

        validator = CompositeValidator(
            [
                ColumnValidator(
                    required_columns=["step"],
                    column_types={"step": int},
                ),
                DataQualityValidator(min_rows=1),
            ]
        )
        self.set_validator(validator)

    def register_functions(self) -> None:
        self._functions = {
            "run_foundation_analyses": make_analysis_function(run_foundation_analyses),
            "analyze_synthesis": make_analysis_function(analyze_system_dynamics_synthesis),
            "write_unified_report": make_analysis_function(write_unified_system_dynamics_report),
        }

        synth = [
            self._functions["analyze_synthesis"],
            self._functions["write_unified_report"],
        ]
        self._groups = {
            "all": [
                self._functions["run_foundation_analyses"],
                *synth,
            ],
            "synthesis": synth,
            "foundation": [self._functions["run_foundation_analyses"]],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        return SimpleDataProcessor(process_system_dynamics_data)


system_dynamics_module = SystemDynamicsModule()
