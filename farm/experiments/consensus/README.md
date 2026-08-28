# Political consensus experiment

A reproducible social-choice simulation comparing political selection paradigms on one
concrete task: after an election, a single steward allocates a fixed budget of 1.0 across
five projects that help different slices of the population.

## What the default cell can answer

**Primary question (pre-registered):** holding λ's marginal distribution fixed, and
holding the *set of people* fixed, do individual-centered ballot formats
(`individual`, `score`, `latent_match`) change the winner's allocation in a way
that raises **minority-cluster welfare** and/or **total welfare** relative to
`party`?

This is a selection-rule comparison with **exogenous types**. It is not a test of
loyalty formation. Voters never see λ in the default generator
(`λ ~ Beta(2.2, 2.2)` independent of platform), so `E[λ_winner]` is the Beta mean
under every rule by construction. The report does not treat a flat λ profile as
an empirical finding.

`--lambda-correlated` rank-couples high λ to extreme platforms. That condition
can make `E[λ_winner]` differ. It is a **robustness appendix**, named in
`run_config.json` as `primary_question = lambda_selection_robustness`. It is not
the default.

`--mechanism reelection` is the incentive cell: winners *choose* λ to maximize
the re-election rate among a random observation sample (an observer retains if
their utility is at least the λ=0 allocation) plus a weight on loyal targeting.
Party vs individual can then differ through coalition size. See `mechanism.py`
for the theories that cell can and cannot distinguish.

## Decision problem

- N voters, M candidates, P = 5 projects, budget = 1.0.
- Projects (honest generator tags — not winner-relative):
  `public_good`, `majority_pork`, `minority_pork`, `prestige`, `periphery_buffer`.
  `periphery_buffer` is raised for voters far from the population center; that is
  not the same as electoral-loser status.
- Voter `i` has latent `prefs[i] ∈ R^5` and nonnegative `benefits[i]` summing to 1;
  utility of allocation `a` is `benefits[i] · a`.
- Candidate `j` has platform `cplat[j] ∈ R^5` and loyalty `λ[j] ∈ [0, 1]`
  (`λ = 1`: directed component steers toward supporters' mean benefit direction; `λ = 0`: directed component steers toward everyone's mean benefit direction).
- Winner's allocation: `directed = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`,
  then `raw = 0.72·directed + 0.28·simplex(clip(platform, 0, ∞))`. The platform is
  renormalized onto the simplex *before* mixing so the weights are a convex
  combination of matched units. Zero supporters ⇒ all-voter direction only.

## Paradigms

Selection treatments (peer rows):

1. `party` — two party brands at cluster means; nearest-party voting; each party
   nominates its closest candidate; supporters are the winning party's voters.
2. `individual` — nearest-candidate plurality; supporters voted for the winner.
3. `score` — 0–10 scores from inverted distance; highest mean score wins; supporters
   top-scored the winner.
4. `latent_match` — the candidate closest to the mean of all privately submitted
   preference vectors wins; supporters are voters whose nearest candidate is the winner.

Not a voting rule:

5. `constrained_individual` (behind `--include-constrained`) — same election as
   `individual`, but the winner is forced to `λ_effective = min(λ, λ_cap)`
   (default cap 0.25). A constitutional-duty contrast. Excluded from hypothesis
   loops and from "which rule wins" language.

Allocation baselines (same draw; not selection treatments): `random_winner`,
`utilitarian` (simplex vertex maximizing mean utility), `egalitarian` (maximin).

`--voting abandon_trailing` applies a Duverger-style heuristic to plurality
(`individual` / the cap overlay). Sincere plurality remains the default baseline.

## Multiple-comparison policy

Per (population, n_candidates) cell, primary endpoints are paired differences
versus `party` on the same trial index:

1. Δ `minority_welfare` (fixed generator / PCA partition)
2. Δ `total_welfare`

for each of `individual`, `score`, `latent_match` (six tests). When the cell is
`--lambda-correlated` or `--mechanism reelection`, Δ `lambda_winner` is added.

Inference: Wilcoxon signed-rank (two-sided) on paired differences; 95% Student-t
CI; paired Cohen's d. **Holm** step-down FWER correction across the primary
family. Everything else (election-endogenous loser welfare, gap, Gini, …) is
exploratory and unadjusted.

Written to `contrasts.csv` and quoted in `REPORT.md`.

## Usage

```bash
# default experiment (primary cell)
python run_experiment.py --trials 250 --voters 400 --candidates 8 \
    --population two_cluster --seed 0 --out results/consensus

# robustness appendix: λ rank-coupled to platform extremity
python run_experiment.py --lambda-correlated --include-constrained \
    --out results/consensus_lambda_correlated

# incentive cell: winners choose λ
python run_experiment.py --mechanism reelection --out results/consensus_reelection

# sweep across populations and candidate counts
python run_experiment.py sweep --trials 100 \
    --populations two_cluster,three_cluster,rural_town --candidates 6,8,12

# social-media MP4 of one trial's dynamic, rendered from live simulation data (needs ffmpeg)
python run_experiment.py animate --seed 0 --trial 0 \
    --out results/consensus_media/consensus_dynamics.mp4

# produced audience explainer, numbers loaded from run outputs (needs `pip install manim`)
python run_experiment.py overview --results results/consensus \
    --correlated-results results/consensus_lambda_correlated \
    --out results/consensus_media/consensus_overview.mp4
```

The two video commands serve different purposes: `animate` (matplotlib) renders the raw
dynamic of an actual seeded trial, while `overview` (manim, optional dependency) builds a
presentation-style explainer of the experiment and its headline results, reading every
displayed number from the given runs' `summary.csv` files. Use `--quality preview` with
`overview` for fast iteration.

Population types: `one_cluster`, `two_cluster` (default), `three_cluster`,
`rural_town` (70/30 cluster sizes). Extra flags: `--include-constrained`,
`--lambda-cap`, `--lambda-correlated` (robustness appendix), `--mechanism`,
`--voting`, `--no-persist-ballots`.

Outputs in `--out`: `trials.csv` (one row per paradigm × trial, plus baseline
rows), `summary.csv`, `allocation_means.csv`, `contrasts.csv`, `run_config.json`
(the exact command with a `{run_dir}` placeholder, plus the parsed config,
including `primary_question`), `figures/`, and `REPORT.md`. Synthetic ballots,
supporter masks, and cluster ids go under `private/` (default on), including
sweeps (cells stacked in cell-major order). They are an audit trail for
synthetic voters, not a privacy claim, and they are not notarized. Official
record: aggregates only.

`python run_experiment.py verify-report --results <dir>` recomputes
`summary.csv`, `allocation_means.csv`, `contrasts.csv`, and `REPORT.md` from
`trials.csv` and byte-compares them against the files on disk.

## Metrics

Per trial: fixed-partition `majority_welfare` / `minority_welfare` /
`cluster_k_welfare`; distributional `min_utility`, `p10_utility`, `gini_utility`;
normalized `(metric − random) / (utilitarian − random)` for total and minority
welfare; election-endogenous `supporter_welfare`, `loser_welfare`, `gap`,
`loser_share` (labeled as such); `lambda_winner`, `lambda_effective`; the
winner's 5-vector allocation.

`rural_town` party `loser_share` equals the minority bloc size by construction
and is not a treatment effect.

## Tests

Invariant and property tests live in `tests/experiments/test_consensus_invariants.py`:
smoke tests (simplex allocations, finite utilities, seed identity, λ cap) plus
monotonicity of complement welfare in λ, in-family total-welfare at λ=0 vs λ=1,
platform scale invariance, metamorphic relabeling, and paired-contrast CIs.

```bash
pytest tests/experiments/test_consensus_invariants.py
```

## Module layout

| Module | Role |
|--------|------|
| `population.py` | Voter/candidate generation (clusters, benefits, λ) |
| `allocation.py` | Steward's budget allocation rule |
| `paradigms.py` | Election rules and supporter definitions |
| `mechanism.py` | Re-election λ choice (incentive cell) |
| `metrics.py` | Per-trial welfare, baselines, tails |
| `contrasts.py` | Paired Δ, CI, Wilcoxon, Holm, effect size |
| `experiment.py` | Trial orchestration, summaries, artifact writing |
| `plots.py` | Matplotlib figures (Agg) |
| `report.py` | Auto-generated `REPORT.md` |
| `animate.py` | MP4 of one trial's dynamic (matplotlib + ffmpeg) |
| `overview_video.py` | Produced audience explainer (manim, optional dep) |

The scaffold runner `farm/runners/consensus_paradigms_experiment.py` is a thin
wrapper over `run_trials`.
