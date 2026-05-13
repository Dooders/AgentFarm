# Balanced Profile as a Transition-Regime Workflow

This workflow turns the balanced-profile hypothesis into an evidence-gated
experiment.  The goal is not to force a narrative out of noisy seed sweeps; the
goal is to decide whether the project can honestly write a paragraph of the
form:

> In profile P, with parameter Q in range R, the system transitions between
> mode A and mode B with probability p, controlled by mechanism M.

If the evidence is insufficient, the analyzer says so and lists the missing
gates.

## Runner

Use `scripts/run_transition_regime_experiment.py` to generate a factorized
matrix around the stable balanced profile.

```bash
python scripts/run_transition_regime_experiment.py \
  --resource-levels 9 9.5 10 10.5 11 \
  --seeds 42 7 19 101 137 256 \
  --output-dir experiments/transition_regime
```

By default the runner varies `initial_agent_resource_level` and couples the
rest of the resource buffer along the line that recovers the existing stable
sub-profiles:

| initial agent resources | resource nodes | regen rate |
| ---: | ---: | ---: |
| 8 | 32 | 0.14 |
| 10 | 34 | 0.15 |
| 12 | 36 | 0.16 |

That coupling can be disabled with `--no-couple-resource-buffer-line` if you
want to vary only the agents' starting resources while holding environment
supply fixed.

### Mechanism interventions

Add interventions explicitly:

```bash
python scripts/run_transition_regime_experiment.py \
  --interventions baseline crossover_on \
  --resource-levels 9.5 10 10.5 \
  --seeds 42 7 19 101 137 256 \
  --output-dir experiments/transition_regime_crossover
```

- `baseline`: mutation-only inheritance, matching the stable-profile seed
  sweep.
- `crossover_on`: enables crossover to test whether gene flow smooths or
  collapses the balanced transition boundary.
- `long_horizon`: runs the configured long-horizon resource levels (default
  `10.0`) for `--long-horizon-num-steps` steps.

Always inspect the matrix first for large runs:

```bash
python scripts/run_transition_regime_experiment.py \
  --interventions baseline crossover_on \
  --resource-levels 9.5 10 10.5 \
  --seeds 42 7 \
  --dry-run
```

## Analyzer

After runs complete:

```bash
python scripts/analyze_transition_regime.py \
  --sweep-dir experiments/transition_regime
```

The analyzer writes:

- `transition_regime_summary.json`
- `transition_regime_summary.md`
- `mode_assignments.csv`
- `transition_probability_by_parameter.csv`
- `mode_assignments_vs_parameter.png`
- `transition_probability_by_parameter.png`
- `mechanism_evidence.png`
- `exit_paragraph.txt` only when the evidence gates pass.

## Evidence gates

The default gates are intentionally conservative:

1. At least two detected modes must each have at least two supporting runs.
2. At least one parameter value/range must have at least six baseline runs.
3. The transition probability in that value/range must be non-degenerate
   (`0.2 ≤ p ≤ 0.8` by default).
4. At least one candidate mechanism must clear the effect threshold.

If any gate fails, the Markdown summary lists the reason and no exit paragraph
is produced.

## Mode and mechanism interpretation

Modes are currently labelled by final/late speciation outcome:

- `low_speciation`
- `high_speciation`

The default classifier uses run-level features from intrinsic telemetry:

- late-window speciation mean,
- late-window speciation slope,
- early population overshoot.

When too few complete runs exist for a two-component model, the analyzer falls
back to a transparent threshold classifier.

Candidate mechanisms are evaluated from existing telemetry:

- **gene_flow**: crossover-on changes mode entropy or final-speciation
  variance relative to baseline.
- **selection_strength**: high/low modes differ in late effective
  selection-strength telemetry.
- **startup_transient**: high/low modes differ in early population overshoot.

These mechanism labels are hypotheses supported by the telemetry contrast, not
proof of causality.  The exit paragraph should still be read as an empirical
summary of the configured matrix.

## Fast smoke check

For code validation rather than scientific evidence:

```bash
python scripts/run_transition_regime_experiment.py \
  --resource-levels 10 \
  --seeds 42 \
  --num-steps 5 \
  --warmup-steps 0 \
  --snapshot-interval 1 \
  --output-dir /tmp/agentfarm_transition_smoke

python scripts/analyze_transition_regime.py \
  --sweep-dir /tmp/agentfarm_transition_smoke
```

The smoke analyzer should refuse to generate `exit_paragraph.txt`; one run is
not enough evidence for a transition-regime claim.
