# Political consensus experiment

A reproducible social-choice simulation comparing political selection paradigms on one
concrete task: after an election, a single steward allocates a fixed budget of 1.0 across
five projects that help different slices of the population. The scientific question is
whether individual-centered selection (no parties) produces winners who treat
non-supporters better than party selection does.

This package fills the capability gap tracked as "social-choice / election simulation
with budget-allocation experiments" — AgentFarm previously had no voting rules, election
mechanics, or budget-allocation models.

## Decision problem

- N voters, M candidates, P = 5 projects, budget = 1.0.
- Projects: `core_services` (small benefit to almost everyone), `coalition_club`
  (concentrated benefits), `outgroup_repair` (helps people unlike the winner),
  `prestige_project` (visible, weak broad value), `buffer_reserve` (insurance that
  mainly protects electoral losers).
- Voter `i` has latent `prefs[i] ∈ R^5` and nonnegative `benefits[i]` summing to 1;
  utility of allocation `a` is `benefits[i] · a`.
- Candidate `j` has platform `cplat[j] ∈ R^5` and loyalty `λ[j] ∈ [0, 1]`
  (`λ = 1`: directed component steers toward supporters' mean benefit direction; `λ = 0`: directed component steers toward everyone's mean benefit direction).
- Winner's allocation: `raw = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`,
  then `raw = 0.72·raw + 0.28·clip(platform, 0, ∞)`, normalized. Zero supporters ⇒
  all-voter direction only.

## Paradigms

1. `party` — two party brands at cluster means; nearest-party voting; each party
   nominates its closest candidate; supporters are the winning party's voters.
2. `individual` — nearest-candidate plurality; supporters voted for the winner.
3. `score` — 0–10 scores from inverted distance; highest mean score wins; supporters
   top-scored the winner.
4. `latent_match` — the candidate closest to the mean of all privately submitted
   preference vectors wins; supporters are voters whose nearest candidate is the winner.
5. `constrained_individual` (behind `--include-constrained`) — same election as
   `individual`, but the winner is forced to `λ_effective = min(λ, λ_cap)`
   (default cap 0.25). A constitutional-duty contrast, not a voting rule.

## Usage

```bash
# default experiment
python run_experiment.py --trials 250 --voters 400 --candidates 8 \
    --population two_cluster --seed 0 --out results/consensus

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
`--lambda-cap`, `--lambda-correlated` (ties high λ to cluster-extreme platforms).

Outputs in `--out`: `trials.csv` (one row per paradigm × trial), `summary.csv`,
`allocation_means.csv`, `run_config.json` (the exact command with a `{run_dir}`
placeholder, plus the parsed config), `figures/{welfare_by_paradigm,
gap_vs_loser_share,lambda_by_paradigm}.png`, and `REPORT.md` auto-written from the
just-run numbers. Only aggregates and winner allocations are persisted — no
individual ballots.

`python run_experiment.py verify-report --results <dir>` recomputes
`summary.csv`, `allocation_means.csv`, and `REPORT.md` from `trials.csv` and
byte-compares them against the files on disk, so the derived artifacts are
verified to follow from the raw trial data.

## Metrics

Per trial and aggregated: `total_welfare`, `supporter_welfare`, `loser_welfare`,
`gap`, `lambda_winner` (pre-cap), `lambda_effective`, `loser_share`, the winner's
5-vector allocation, winner id, paradigm, seed, and population type.

## Tests

Invariant tests live in `tests/experiments/test_consensus_invariants.py`:
allocations sum to 1 ± 1e-9 and are nonnegative, utilities are finite, party
loser share is near 0.5 under two equal clusters, identical seed + params give an
identical summary, and `constrained_individual` never allocates with λ above the cap.

```bash
pytest tests/experiments/test_consensus_invariants.py
```

## Module layout

| Module | Role |
|--------|------|
| `population.py` | Voter/candidate generation (clusters, benefits, λ) |
| `allocation.py` | Steward's budget allocation rule |
| `paradigms.py` | Election rules and supporter definitions |
| `metrics.py` | Per-trial welfare metrics |
| `experiment.py` | Trial orchestration, summaries, artifact writing |
| `plots.py` | Matplotlib figures (Agg) |
| `report.py` | Auto-generated `REPORT.md` |
| `animate.py` | MP4 of one trial's dynamic (matplotlib + ffmpeg) |
| `overview_video.py` | Produced audience explainer (manim, optional dep) |
