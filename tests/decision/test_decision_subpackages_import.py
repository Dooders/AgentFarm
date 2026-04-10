"""Import smoke tests for decision subpackages (keeps package __init__ lines covered)."""

import unittest


class TestDecisionSubpackagesImport(unittest.TestCase):
    def test_benchmark_package_import(self):
        import farm.core.decision.benchmark as benchmark_pkg

        self.assertEqual(benchmark_pkg.__all__, ["AlgorithmBenchmark"])

    def test_training_package_import(self):
        import farm.core.decision.training as training_pkg

        self.assertEqual(
            set(training_pkg.__all__),
            {
                "AlgorithmTrainer",
                "CROSSOVER_MODES",
                "ChildArchitectureSpec",
                "CrossoverRecipe",
                "DistillationConfig",
                "DistillationMetrics",
                "DistillationTrainer",
                "EpisodeEnvProtocol",
                "ExperienceCollector",
                "FINETUNE_OPTIMIZERS",
                "FineTuneRegime",
                "FineTuner",
                "FineTuningConfig",
                "FineTuningMetrics",
                "LEADERBOARD_COLUMNS",
                "ManifestEntry",
                "PairwiseComparison",
                "PolicyRolloutAdapter",
                "PostTrainingQuantizer",
                "QATConfig",
                "QATMetrics",
                "QATTrainer",
                "QUANTIZATION_APPLIED_MODES",
                "QuantizationConfig",
                "QuantizationResult",
                "QuantizedValidationReport",
                "QuantizedValidationThresholds",
                "QuantizedValidator",
                "REPORT_SCHEMA_VERSION",
                "RecombinationEvaluator",
                "RecombinationReport",
                "RecombinationThresholds",
                "SearchConfig",
                "RolloutComparisonResult",
                "SeededLinearMDP",
                "SimEpisodeStats",
                "SimRolloutConfig",
                "SimRolloutResult",
                "StudentValidator",
                "ValidationReport",
                "ValidationThresholds",
                "WeightOnlyFakeQuantLinear",
                "build_finetune_optimizer",
                "build_leaderboard",
                "compare_outputs",
                "compare_parent_student_rollouts",
                "relative_return_drop",
                "crossover_checkpoints",
                "crossover_quantized_state_dict",
                "generate_recommendation",
                "initialize_child_from_crossover",
                "load_finetuning_config_from_yaml",
                "load_qat_checkpoint",
                "load_quantized_checkpoint",
                "make_shifted_states",
                "run_crossover_search",
                "SHIFT_TYPES",
                "split_replay_buffer",
                "apply_gaussian_noise",
                "apply_input_scaling",
            },
        )


if __name__ == "__main__":
    unittest.main()
