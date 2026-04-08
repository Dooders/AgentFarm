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
                "DistillationConfig",
                "DistillationMetrics",
                "DistillationTrainer",
                "ExperienceCollector",
                "FineTuner",
                "FineTuningConfig",
                "FineTuningMetrics",
                "PostTrainingQuantizer",
                "QATConfig",
                "QATMetrics",
                "QATTrainer",
                "QuantizationConfig",
                "QuantizationResult",
                "QuantizedValidationReport",
                "QuantizedValidationThresholds",
                "QuantizedValidator",
                "StudentValidator",
                "ValidationReport",
                "ValidationThresholds",
                "WeightOnlyFakeQuantLinear",
                "compare_outputs",
                "crossover_checkpoints",
                "crossover_quantized_state_dict",
                "initialize_child_from_crossover",
                "load_qat_checkpoint",
                "load_quantized_checkpoint",
            },
        )


if __name__ == "__main__":
    unittest.main()
