#!/usr/bin/env bash
# Phase 2 docs restructure: moves only (no automatic link rewriting).
set -euo pipefail
cd "$(dirname "$0")/.."
DOCS=docs

move_dir() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -d "$dst" ]]; then rm -rf "$dst"; fi
  cp -a "$src" "$dst"
  git add "$dst"
  git rm -r "$src"
  echo "Moved $src -> $dst"
}

mkdir -p "$DOCS"/{concepts,reference,reference/analysis,research,archive/analysis-extensions,guides}

# --- root file moves (git mv) ---
declare -A FILES=(
  [ExperimentQuickStart.md]=getting-started/experiments-quickstart.md
  [LOGGING_QUICK_REFERENCE.md]=reference/logging-quick-reference.md
  [PROFILING_AND_BENCHMARKING_PLAN.md]=archive/profiling-and-benchmarking-plan.md
  [action_data.md]=concepts/action-data.md
  [action_system.md]=concepts/actions.md
  [agents.md]=concepts/agents-and-decisions.md
  [alien_invasion_combat_simulation.md]=research/alien-invasion-combat-simulation.md
  [api_reference.md]=reference/api-reference.md
  [architectural_recommendations.md]=design/architectural-recommendations.md
  [core_architecture.md]=concepts/core-architecture.md
  [deep_q_learning.md]=concepts/deep-q-learning.md
  [dependency-injection.md]=concepts/dependency-injection.md
  [deployment.md]=guides/deployment.md
  [deterministic_simulations.md]=guides/deterministic-simulations.md
  [distillation_soft_label_comparison.md]=research/distillation-soft-label-comparison.md
  [dynamic_channel_system.md]=concepts/dynamic-channel-system.md
  [evolution_surface_catalog.md]=research/evolution-surface-catalog.md
  [experiment_analysis.md]=guides/experiment-analysis.md
  [experiment_runner.md]=guides/experiment-runner.md
  [experiments.md]=research/experiments-catalog.md
  [generic_combat_simulation_howto.md]=guides/generic-combat-simulation.md
  [generic_simulation_scenario_howto.md]=guides/generic-simulation-scenario.md
  [genetics_analysis.md]=guides/genetics-analysis.md
  [glossary.md]=reference/glossary.md
  [health_resource_analysis.md]=guides/health-resource-analysis.md
  [identity.md]=concepts/identity.md
  [initial_diversity.md]=concepts/initial-diversity.md
  [ipc-api.md]=reference/ipc-api.md
  [logging_guide.md]=guides/logging.md
  [memmap_optimization.md]=concepts/memmap-optimization.md
  [metrics.md]=reference/metrics.md
  [module_overview.md]=concepts/module-overview.md
  [monitoring.md]=guides/monitoring.md
  [observation_channels.md]=concepts/observation-channels.md
  [perception_system_design.md]=concepts/perception-system-design.md
  [perception_system.md]=concepts/perception-system.md
  [redis_agent_memory.md]=concepts/redis-agent-memory.md
  [repositories.md]=reference/repositories-overview.md
  [state_system.md]=concepts/state-system.md
  [STREAMING_CHUNKING.md]=concepts/streaming-chunking.md
  [usage_examples.md]=guides/usage-examples.md
  [howto/neural_recombination_runbook.md]=guides/neural-recombination-runbook.md
)

for old in "${!FILES[@]}"; do
  new="${FILES[$old]}"
  if [[ -f "$DOCS/$old" ]]; then
    git mv "$DOCS/$old" "$DOCS/$new"
  fi
done

# --- directory moves ---
for pair in \
  "config:reference/config" \
  "data:reference/data" \
  "api:reference/api" \
  "devlog:research/devlog" \
  "experiments:research/experiments" \
  "features:archive/features" \
  "refactoring:archive/refactoring" \
  "spatial:concepts/spatial" \
  "electron:reference/electron"; do
  src="${pair%%:*}"
  dst="${pair##*:}"
  if [[ -d "$DOCS/$src" ]]; then
    move_dir "$DOCS/$src" "$DOCS/$dst"
  fi
done

# --- analysis split ---
ARCHIVE=(
  AGENTS_LEARNING_RESOURCES_EXTENSIONS.md
  ANALYSIS_EXTENSIONS_MASTER_INDEX.md
  ANALYSIS_EXTENSIONS_QUICK_REFERENCE.md
  COMPLETE_ANALYSIS_ENHANCEMENT_SUMMARY.md
  GENESIS_COMBAT_ACTIONS_EXTENSIONS.md
  OPTIMIZATION_IMPLEMENTATION_NOTES.md
  POPULATION_ANALYSIS_EXTENSIONS.md
  POPULATION_ANALYSIS_OPTIMIZATION_SUMMARY.md
  POPULATION_ANALYSIS_QUICK_START.md
  POPULATION_ANALYSIS_ROADMAP.md
  README_POPULATION_ENHANCEMENTS.md
  SPATIAL_ANALYSIS_EXTENSIONS.md
  TEMPORAL_SOCIAL_COMPARATIVE_EXTENSIONS.md
)
KEEP=(Advantage.md Dominance.md Genesis.md Social.md)

if [[ -d "$DOCS/analysis/modules" ]]; then
  move_dir "$DOCS/analysis/modules" "$DOCS/reference/analysis/modules"
fi
for f in "${ARCHIVE[@]}"; do
  if [[ -f "$DOCS/analysis/$f" ]]; then
    git mv "$DOCS/analysis/$f" "$DOCS/archive/analysis-extensions/$f"
  fi
done
for f in "${KEEP[@]}"; do
  if [[ -f "$DOCS/analysis/$f" ]]; then
    git mv "$DOCS/analysis/$f" "$DOCS/reference/analysis/$f"
  fi
done
if [[ -d "$DOCS/analysis" ]] && [[ -z "$(ls -A "$DOCS/analysis" 2>/dev/null)" ]]; then
  rmdir "$DOCS/analysis"
fi
if [[ -d "$DOCS/howto" ]] && [[ -z "$(ls -A "$DOCS/howto" 2>/dev/null)" ]]; then
  rmdir "$DOCS/howto"
fi

# remove merged stubs
git rm -f "$DOCS/user-guide.md" "$DOCS/developer-guide.md" 2>/dev/null || rm -f "$DOCS/user-guide.md" "$DOCS/developer-guide.md"

echo "Phase 2 moves complete."
