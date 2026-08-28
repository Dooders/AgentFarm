---
title: "Consensus experiment: stop peer-ranking constrained_individual; retire the scaffold; drop prototype numbers"
type: Task
labels: [Experiment]
---

## Context

Three process problems still contaminate the result.

### `constrained_individual` is not a voting rule

It reuses `individual` election and caps `λ_effective`. That manipulates the
**outcome function**, not the selection rule. It will trivially raise loser
welfare (for a fixed supporter set) and will be read as if it competed on
equal terms.

The package README already says "constitutional-duty contrast, not a voting
rule." The correlated committed report still prints it as a fifth paradigm
row in the same table, with the same hypothesis machinery available.

### Prototype-orientation paragraph is a confirmation-bias machine

`docs/research/experiments/consensus_paradigms.md` still states expected
numbers ("λ_winner nearly flat… party ~50% losers… non-party ~70–75% loser
share"), then the same system auto-writes `REPORT.md` with hardcoded
thresholds that encode those expectations
(`LAMBDA_UNCHANGED_EPS`, `WELFARE_FLAT_REL_EPS`, `LOSER_SHARE_LARGE_RISE`).

The new README dropped the numbers; the scaffold spec and the verdict
thresholds did not.

### Two implementations

| Path | Role |
|---|---|
| `farm/experiments/consensus/` + `run_experiment.py` | actual experiment, committed results |
| `farm/runners/consensus_paradigms_experiment.py` + `scripts/run_consensus_paradigms_experiment.py` | scaffold: iid-ish `N(0, 0.7)` platforms, clip-normalized prefs, same 0.72/0.28 scale bug, independent λ, no cluster-fixed metrics |

The catalog (`docs/research/experiments-catalog.md`) still points at the
scaffold. Docs say **Status: Scaffold**. Anyone reproducing from the catalog
runs the worse code.

## Goal

1. Keep `constrained_individual` behind `--include-constrained`. Do not
   include it in hypothesis loops, default summaries, or "which rule wins"
   language. If shown, put it in a separate "constitutional cap" section that
   says it is not a selection treatment.
2. Delete the prototype-orientation numbers from
   `docs/research/experiments/consensus_paradigms.md`. Move verdict
   thresholds out of "expected finding" territory — either drop them or
   compute them from the paired contrasts issue.
3. Retire or rebase the scaffold: one implementation. Either delete the
   runner and point the catalog / notary docs at
   `farm.experiments.consensus`, or make the runner a thin wrapper over
   `run_trials`.

## Acceptance

- [ ] `constrained_individual` is absent from the default hypothesis table
      and from "paradigm vs party" verdicts.
- [ ] Scaffold spec no longer publishes expected λ / loser_share numbers.
- [ ] Catalog and notary quick-start run `run_experiment.py` (or a wrapper
      that calls the same code).
- [ ] Either the old runner is gone, or `ConsensusParadigmsExperiment.run`
      calls `farm.experiments.consensus.experiment.run_trials`.

## Files

- `farm/experiments/consensus/report.py`
- `farm/experiments/consensus/experiment.py`
- `docs/research/experiments/consensus_paradigms.md`
- `docs/research/experiments-catalog.md`
- `farm/runners/consensus_paradigms_experiment.py`
- `scripts/run_consensus_paradigms_experiment.py`
- `tests/runners/test_consensus_paradigms_experiment.py`
