"""Training module for ML algorithms."""

from .collector import ExperienceCollector
from .crossover_search import (
    CrossoverRecipe,
    FineTuneRegime,
    LEADERBOARD_COLUMNS,
    ManifestEntry,
    SearchConfig,
    build_leaderboard,
    generate_recommendation,
    run_crossover_search,
)
from .crossover import (
    CROSSOVER_MODES,
    ChildArchitectureSpec,
    crossover_checkpoints,
    crossover_quantized_state_dict,
    initialize_child_from_crossover,
)
from .finetune import (
    QUANTIZATION_APPLIED_MODES,
    FineTuner,
    FineTuningConfig,
    FineTuningMetrics,
)
from .quantize_ptq import (
    PostTrainingQuantizer,
    QuantizationConfig,
    QuantizationResult,
    QuantizedValidationReport,
    QuantizedValidationThresholds,
    QuantizedValidator,
    compare_outputs,
    load_quantized_checkpoint,
)
from .quantize_qat import (
    QATConfig,
    QATMetrics,
    QATTrainer,
    WeightOnlyFakeQuantLinear,
    load_qat_checkpoint,
)
from .recombination_eval import (
    REPORT_SCHEMA_VERSION,
    PairwiseComparison,
    RecombinationEvaluator,
    RecombinationReport,
    RecombinationThresholds,
)
from .trainer import AlgorithmTrainer
from .distillation_rollout import (
    RolloutComparisonResult,
    SeededLinearMDP,
    compare_parent_student_rollouts,
    relative_return_drop,
)
from .trainer_distill import (
    DistillationConfig,
    DistillationMetrics,
    DistillationTrainer,
    StudentValidator,
    ValidationReport,
    ValidationThresholds,
)

__all__ = [
    "AlgorithmTrainer",
    "CROSSOVER_MODES",
    "ChildArchitectureSpec",
    "CrossoverRecipe",
    "DistillationConfig",
    "DistillationMetrics",
    "DistillationTrainer",
    "ExperienceCollector",
    "FineTuneRegime",
    "FineTuner",
    "FineTuningConfig",
    "FineTuningMetrics",
    "LEADERBOARD_COLUMNS",
    "ManifestEntry",
    "PairwiseComparison",
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
    "RolloutComparisonResult",
    "SearchConfig",
    "SeededLinearMDP",
    "StudentValidator",
    "ValidationReport",
    "ValidationThresholds",
    "WeightOnlyFakeQuantLinear",
    "build_leaderboard",
    "compare_outputs",
    "compare_parent_student_rollouts",
    "crossover_checkpoints",
    "crossover_quantized_state_dict",
    "generate_recommendation",
    "initialize_child_from_crossover",
    "load_qat_checkpoint",
    "load_quantized_checkpoint",
    "relative_return_drop",
    "run_crossover_search",
]
