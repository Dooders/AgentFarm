from farm.analysis.dominance.analyze import (
    analyze_dominance_switch_factors,
    analyze_reproduction_dominance_switching,
    process_dominance_data,
)
from farm.analysis.dominance.ml import (
    prepare_features_for_classification,
    run_dominance_classification,
    train_classifier,
)
from farm.analysis.dominance.plot import (
    plot_comprehensive_score_breakdown,
    plot_correlation_matrix,
    plot_dominance_comparison,
    plot_dominance_distribution,
    plot_dominance_switches,
    plot_feature_importance,
    plot_reproduction_vs_dominance,
    plot_resource_proximity_vs_dominance,
)
