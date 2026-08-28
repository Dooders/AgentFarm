---
title: "Consensus experiment: λ is invisible to voters, so E[λ_winner] ≈ 0.5 under every rule"
type: Bug
labels: [Experiment]
---

## Context

The experiment's hypothesis (`farm/experiments/consensus/report.py`,
`docs/research/experiments/consensus_paradigms.md`) is that individual / score
/ latent_match **select lower-λ winners** than party.

No election rule reads `λ`. Default generation is still
`λ ~ Beta(2.2, 2.2)` independent of platform:

```python
# farm/experiments/consensus/population.py
lam = rng.beta(LAMBDA_BETA_A, LAMBDA_BETA_B, size=n_candidates)
# optional: rank-couple to platform extremity if lambda_correlated
```

Voters vote on preference distance. `latent_match` matches platform to mean
prefs. Nothing in any selection rule can condition on loyalty. So
`E[λ_winner] ≈ 0.5` under every treatment, analytically, before any trial
runs.

Default committed results confirm it: max `|Δλ| = 0.0065`, and the
auto-report's "λ_winner unchanged" falsifier is **triggered**. That is
arithmetic, not a finding.

`--lambda-correlated` is a researcher degree of freedom that **flips the core
result**. It rank-couples high λ to extreme platforms, so score / latent_match
(who pick centrists) get `λ ≈ 0.24 / 0.20` vs party `0.43`. Individual still
does not (`0.44`). Nothing pre-registers which condition is primary. Default
= independent = the λ finding is guaranteed.

## Goal

Make the λ-selection claim answerable, or stop asking it.

Pick one primary design (pre-register it in the README and `run_config.json`):

1. **λ observable to voters** — e.g. a costly signal, or platform extremity
   that voters see and that is structurally tied to λ (not an optional flag).
   Then selection rules can condition on loyalty and `E[λ_winner]` can differ.
2. **Change the question** — drop "select lower-λ winners" from the hypothesis
   and report. Ask only whether *ballot format* changes allocation via the
   supporter set and the platform mix, holding λ's marginal fixed.
3. **Endogenous λ** (see the mechanism issue) — re-election / entry so loyalty
   is chosen, not drawn.

Do not leave `--lambda-correlated` as an undocumented switch that turns the
headline on and off. If both conditions are run, name one primary in
`ExperimentConfig` and in the report, and treat the other as a robustness
appendix.

## Acceptance

- [ ] Default hypothesis text matches what the generator can identify.
- [ ] If the λ-selection hypothesis is kept, some voter-visible quantity is
      correlated with λ in the **default** config (not only behind a flag),
      and a unit test fails if that correlation is removed.
- [ ] `--lambda-correlated` is either the documented primary, removed, or
      clearly marked secondary in `REPORT.md` and the overview video.
- [ ] Regenerated default `REPORT.md` does not treat `λ_winner ≈ 0.5` across
      rules as an empirical discovery.

## Files

- `farm/experiments/consensus/population.py` (`generate_candidates`)
- `farm/experiments/consensus/paradigms.py` (no rule reads `lam`)
- `farm/experiments/consensus/report.py` (hypothesis + `LAMBDA_UNCHANGED_EPS`)
- `run_experiment.py` (`--lambda-correlated`)
- `farm/experiments/consensus/README.md`
- `results/consensus/REPORT.md` (regenerate after the fix)
