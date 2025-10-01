#!/usr/bin/env python3
"""
Reproducibility utilities for simulation analysis.

This module provides tools to ensure reproducible analysis results,
including random seed management, environment tracking, and result validation.
"""

import hashlib
import json
import logging
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """Manages reproducibility aspects of analysis."""

    def __init__(self, seed: int = 42):
        """Initialize reproducibility manager.

        Args:
            seed: Random seed for reproducible results
        """
        self.seed = seed
        self.environment_info = self._capture_environment()
        self._set_random_seeds()

    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment information for reproducibility."""
        return {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_executable": sys.executable,
            "working_directory": os.getcwd(),
            "environment_variables": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("PYTHON", "PATH", "CONDA", "VIRTUAL"))
            },
        }

    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducible results."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Set seeds for other libraries if available
        try:
            import torch

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass

        logger.info(f"Random seeds set to {self.seed}")

    def get_environment_info(self) -> Dict[str, Any]:
        """Get captured environment information."""
        return self.environment_info.copy()

    def create_analysis_hash(self, analysis_params: Dict[str, Any]) -> str:
        """Create a hash for analysis parameters to ensure reproducibility.

        Args:
            analysis_params: Dictionary of analysis parameters

        Returns:
            Hash string for the analysis configuration
        """
        # Create a deterministic string from parameters
        param_string = json.dumps(analysis_params, sort_keys=True, default=str)

        # Add environment info
        env_string = json.dumps(self.environment_info, sort_keys=True, default=str)

        # Create hash
        combined_string = f"{param_string}_{env_string}_{self.seed}"
        return hashlib.md5(combined_string.encode()).hexdigest()

    def validate_reproducibility(
        self, results1: Dict[str, Any], results2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that two analysis runs produce identical results.

        Args:
            results1: First analysis results
            results2: Second analysis results

        Returns:
            Validation report
        """
        validation_report = {"identical": True, "differences": [], "summary": {}}

        def compare_values(val1, val2, path=""):
            """Recursively compare values."""
            if type(val1) != type(val2):
                validation_report["identical"] = False
                validation_report["differences"].append(
                    f"{path}: Type mismatch ({type(val1)} vs {type(val2)})"
                )
                return

            if isinstance(val1, dict):
                keys1, keys2 = set(val1.keys()), set(val2.keys())
                if keys1 != keys2:
                    validation_report["identical"] = False
                    validation_report["differences"].append(f"{path}: Key mismatch")
                    return

                for key in keys1:
                    compare_values(
                        val1[key], val2[key], f"{path}.{key}" if path else key
                    )

            elif isinstance(val1, (list, tuple)):
                if len(val1) != len(val2):
                    validation_report["identical"] = False
                    validation_report["differences"].append(
                        f"{path}: Length mismatch ({len(val1)} vs {len(val2)})"
                    )
                    return

                for i, (v1, v2) in enumerate(zip(val1, val2)):
                    compare_values(v1, v2, f"{path}[{i}]")

            elif isinstance(val1, (int, float)):
                if not np.isclose(val1, val2, rtol=1e-10, atol=1e-10):
                    validation_report["identical"] = False
                    validation_report["differences"].append(
                        f"{path}: Value mismatch ({val1} vs {val2})"
                    )

            elif isinstance(val1, str):
                if val1 != val2:
                    validation_report["identical"] = False
                    validation_report["differences"].append(f"{path}: String mismatch")

            else:
                if val1 != val2:
                    validation_report["identical"] = False
                    validation_report["differences"].append(f"{path}: Value mismatch")

        compare_values(results1, results2)

        validation_report["summary"] = {
            "total_differences": len(validation_report["differences"]),
            "validation_passed": validation_report["identical"],
        }

        return validation_report


class AnalysisValidator:
    """Validates analysis results for consistency and correctness."""

    def __init__(self):
        """Initialize analysis validator."""
        self.validation_rules = self._load_validation_rules()

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for different analysis types."""
        return {
            "population_dynamics": {
                "required_keys": ["dataframe", "statistical_analysis", "summary"],
                "dataframe_checks": ["non_empty", "numeric_columns"],
                "statistical_checks": ["confidence_intervals", "p_values_valid"],
            },
            "critical_events": {
                "required_keys": ["step", "agent_type", "p_value", "is_significant"],
                "data_checks": ["p_values_range", "significance_consistency"],
            },
            "agent_interactions": {
                "required_keys": ["interaction_patterns", "statistical_analysis"],
                "statistical_checks": ["chi_square_valid", "interaction_matrix_square"],
            },
            "temporal_patterns": {
                "required_keys": ["time_series_analysis", "cross_correlations"],
                "time_series_checks": ["stationarity_tests", "trend_analysis"],
            },
            "advanced_ml": {
                "required_keys": [
                    "feature_selection",
                    "individual_models",
                    "performance_comparison",
                ],
                "ml_checks": [
                    "model_performance",
                    "feature_importance",
                    "cross_validation",
                ],
            },
        }

    def validate_analysis_result(
        self, analysis_type: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a specific analysis result.

        Args:
            analysis_type: Type of analysis (e.g., 'population_dynamics')
            result: Analysis result to validate

        Returns:
            Validation report
        """
        if analysis_type not in self.validation_rules:
            return {"error": f"Unknown analysis type: {analysis_type}"}

        rules = self.validation_rules[analysis_type]
        validation_report = {
            "analysis_type": analysis_type,
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks_passed": 0,
            "total_checks": 0,
        }

        # Check required keys
        for key in rules.get("required_keys", []):
            validation_report["total_checks"] += 1
            if key not in result:
                validation_report["valid"] = False
                validation_report["errors"].append(f"Missing required key: {key}")
            else:
                validation_report["checks_passed"] += 1

        # Run specific checks
        if "dataframe_checks" in rules:
            validation_report = self._run_dataframe_checks(
                result, rules["dataframe_checks"], validation_report
            )

        if "statistical_checks" in rules:
            validation_report = self._run_statistical_checks(
                result, rules["statistical_checks"], validation_report
            )

        if "data_checks" in rules:
            validation_report = self._run_data_checks(
                result, rules["data_checks"], validation_report
            )

        if "time_series_checks" in rules:
            validation_report = self._run_time_series_checks(
                result, rules["time_series_checks"], validation_report
            )

        if "ml_checks" in rules:
            validation_report = self._run_ml_checks(
                result, rules["ml_checks"], validation_report
            )

        return validation_report

    def _run_dataframe_checks(
        self,
        result: Dict[str, Any],
        checks: List[str],
        validation_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run dataframe-specific validation checks."""
        if "dataframe" not in result:
            return validation_report

        df = result["dataframe"]

        for check in checks:
            validation_report["total_checks"] += 1

            if check == "non_empty":
                if len(df) == 0:
                    validation_report["valid"] = False
                    validation_report["errors"].append("DataFrame is empty")
                else:
                    validation_report["checks_passed"] += 1

            elif check == "numeric_columns":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    validation_report["warnings"].append("No numeric columns found")
                validation_report["checks_passed"] += 1

        return validation_report

    def _run_statistical_checks(
        self,
        result: Dict[str, Any],
        checks: List[str],
        validation_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run statistical validation checks."""
        for check in checks:
            validation_report["total_checks"] += 1

            if check == "confidence_intervals":
                if "statistical_analysis" in result:
                    stats = result["statistical_analysis"]
                    if "confidence_intervals" in stats:
                        validation_report["checks_passed"] += 1
                    else:
                        validation_report["warnings"].append(
                            "No confidence intervals found"
                        )
                else:
                    validation_report["warnings"].append(
                        "No statistical analysis found"
                    )

            elif check == "p_values_valid":
                # Check if p-values are in valid range [0, 1]
                p_values = self._extract_p_values(result)
                invalid_p_values = [p for p in p_values if not (0 <= p <= 1)]
                if invalid_p_values:
                    validation_report["valid"] = False
                    validation_report["errors"].append(
                        f"Invalid p-values found: {invalid_p_values}"
                    )
                else:
                    validation_report["checks_passed"] += 1

        return validation_report

    def _run_data_checks(
        self,
        result: Dict[str, Any],
        checks: List[str],
        validation_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run data-specific validation checks."""
        for check in checks:
            validation_report["total_checks"] += 1

            if check == "p_values_range":
                p_values = self._extract_p_values(result)
                if all(0 <= p <= 1 for p in p_values):
                    validation_report["checks_passed"] += 1
                else:
                    validation_report["valid"] = False
                    validation_report["errors"].append(
                        "P-values outside valid range [0, 1]"
                    )

            elif check == "significance_consistency":
                # Check if significance flags match p-values
                if isinstance(result, list):
                    for item in result:
                        if "p_value" in item and "is_significant" in item:
                            expected_significance = item["p_value"] < 0.05
                            if item["is_significant"] != expected_significance:
                                validation_report["warnings"].append(
                                    f"Inconsistent significance flag for p={item['p_value']}"
                                )
                    validation_report["checks_passed"] += 1

        return validation_report

    def _run_time_series_checks(
        self,
        result: Dict[str, Any],
        checks: List[str],
        validation_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run time series validation checks."""
        for check in checks:
            validation_report["total_checks"] += 1

            if check == "stationarity_tests":
                if "time_series_analysis" in result:
                    ts_analysis = result["time_series_analysis"]
                    has_stationarity = any(
                        "stationarity" in series_data
                        for series_data in ts_analysis.values()
                    )
                    if has_stationarity:
                        validation_report["checks_passed"] += 1
                    else:
                        validation_report["warnings"].append(
                            "No stationarity tests found"
                        )
                else:
                    validation_report["warnings"].append(
                        "No time series analysis found"
                    )

            elif check == "trend_analysis":
                if "time_series_analysis" in result:
                    ts_analysis = result["time_series_analysis"]
                    has_trend = any(
                        "trend" in series_data for series_data in ts_analysis.values()
                    )
                    if has_trend:
                        validation_report["checks_passed"] += 1
                    else:
                        validation_report["warnings"].append("No trend analysis found")

        return validation_report

    def _run_ml_checks(
        self,
        result: Dict[str, Any],
        checks: List[str],
        validation_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run machine learning validation checks."""
        for check in checks:
            validation_report["total_checks"] += 1

            if check == "model_performance":
                if "performance_comparison" in result:
                    perf = result["performance_comparison"]
                    if len(perf) > 0:
                        validation_report["checks_passed"] += 1
                    else:
                        validation_report["warnings"].append(
                            "No model performance data"
                        )
                else:
                    validation_report["warnings"].append(
                        "No performance comparison found"
                    )

            elif check == "feature_importance":
                if "individual_models" in result:
                    models = result["individual_models"]
                    has_importance = any(
                        "feature_importance" in model_data
                        for model_data in models.values()
                        if isinstance(model_data, dict)
                    )
                    if has_importance:
                        validation_report["checks_passed"] += 1
                    else:
                        validation_report["warnings"].append(
                            "No feature importance data"
                        )

            elif check == "cross_validation":
                if "individual_models" in result:
                    models = result["individual_models"]
                    has_cv = any(
                        "cv_scores" in model_data
                        for model_data in models.values()
                        if isinstance(model_data, dict)
                    )
                    if has_cv:
                        validation_report["checks_passed"] += 1
                    else:
                        validation_report["warnings"].append("No cross-validation data")

        return validation_report

    def _extract_p_values(self, result: Dict[str, Any]) -> List[float]:
        """Extract all p-values from analysis result."""
        p_values = []

        def extract_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "p_value" and isinstance(value, (int, float)):
                        p_values.append(value)
                    else:
                        extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)

        extract_recursive(result)
        return p_values

    def validate_complete_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete analysis results.

        Args:
            results: Complete analysis results

        Returns:
            Comprehensive validation report
        """
        validation_report = {
            "overall_valid": True,
            "analysis_validations": {},
            "summary": {
                "total_analyses": 0,
                "valid_analyses": 0,
                "total_checks": 0,
                "passed_checks": 0,
            },
        }

        # Define analysis types to validate
        analysis_types = [
            "population_dynamics",
            "critical_events",
            "agent_interactions",
            "temporal_patterns",
            "advanced_ml",
        ]

        for analysis_type in analysis_types:
            if analysis_type in results:
                validation_report["summary"]["total_analyses"] += 1

                analysis_validation = self.validate_analysis_result(
                    analysis_type, results[analysis_type]
                )

                validation_report["analysis_validations"][
                    analysis_type
                ] = analysis_validation

                if analysis_validation["valid"]:
                    validation_report["summary"]["valid_analyses"] += 1
                else:
                    validation_report["overall_valid"] = False

                validation_report["summary"]["total_checks"] += analysis_validation[
                    "total_checks"
                ]
                validation_report["summary"]["passed_checks"] += analysis_validation[
                    "checks_passed"
                ]

        # Calculate success rate
        if validation_report["summary"]["total_checks"] > 0:
            validation_report["summary"]["success_rate"] = (
                validation_report["summary"]["passed_checks"]
                / validation_report["summary"]["total_checks"]
            )
        else:
            validation_report["summary"]["success_rate"] = 0

        return validation_report


def create_reproducibility_report(
    analysis_params: Dict[str, Any],
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """Create a comprehensive reproducibility report.

    Args:
        analysis_params: Parameters used for analysis
        results: Analysis results
        output_path: Path to save the report (optional)

    Returns:
        Path to the created report
    """
    if output_path is None:
        output_path = Path("reproducibility_report.json")

    # Initialize reproducibility manager
    repro_manager = ReproducibilityManager()

    # Initialize validator
    validator = AnalysisValidator()

    # Create report
    report = {
        "report_metadata": {
            "created_at": datetime.now().isoformat(),
            "analysis_hash": repro_manager.create_analysis_hash(analysis_params),
            "random_seed": repro_manager.seed,
        },
        "environment_info": repro_manager.get_environment_info(),
        "analysis_parameters": analysis_params,
        "validation_results": validator.validate_complete_analysis(results),
        "reproducibility_guidelines": {
            "random_seed": repro_manager.seed,
            "python_version": sys.version,
            "required_packages": [
                "numpy",
                "pandas",
                "scipy",
                "matplotlib",
                "seaborn",
                "sklearn",
                "statsmodels",
                "sqlalchemy",
            ],
            "analysis_version": "Phase 2 - Statistical Enhancement",
        },
    }

    # Save report
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Reproducibility report saved to {output_path}")
    return output_path
