---
title: "Consensus experiment: endogenous loyalty, candidate entry, and non-sincere voting"
type: Feature
labels: [Experiment]
---

## Context

The real difference between party and individual systems runs through
candidate entry, platform positioning, and re-election incentives. In the
current design:

- candidate platforms and λ are drawn from the **same** distribution under
  every paradigm
- there is one election and no future
- loyalty to supporters is exogenous in a model whose question is what makes
  loyalty to supporters vary

Citizen-candidate models (Osborne–Slivinski, Besley–Coate) and the
core-voter / swing-voter literature (Cox–McCubbins vs Lindbeck–Weibull) make
opposing predictions here. This design cannot adjudicate between them. The
limitations blurb already admits λ is exogenous; the hypothesis still reads
as if selection *causes* lower λ.

Separately, all voters are sincere. Plurality with 8 sincere-voting
candidates is the known-worst case for plurality. No strategy, no
abstention, no turnout asymmetry.

This is the large follow-up. The other drafts make the *current* question
well-posed. This one changes the question to one political economy can
actually use.

## Goal

Add a second stage (or a second experiment cell) in which λ and/or platforms
are chosen, not drawn.

Minimum viable mechanism (pick one and document it):

1. **Re-election.** After allocation, a random subset of voters observes
   their utility and votes again. Candidates who will stand again choose λ
   (and optionally platform) to maximize re-election plus a weight on
   ideological / loyal payoff. Party vs individual then differ through the
   re-election constraint (larger vs smaller winning coalition).
2. **Citizen-candidate entry.** A pool of potential candidates decides
   whether to enter at a cost. Party nomination vs free entry produces
   different platform/λ equilibria.
3. **Core vs swing targeting.** Replace the λ interpolation with an explicit
   targeting rule (core supporters vs swing voters) whose optimum depends on
   the electoral rule.

And, independently or in the same design:

- allow strategic voting under plurality (e.g. Duverger-style best-response
  or a simple "abandon trailing candidates" heuristic)
- allow abstention as a function of distance / pivotality

Do not claim the one-shot exogenous-λ model tests those theories.

## Acceptance

- [ ] A documented mechanism (re-election, entry, or targeting) exists as a
      first-class population / paradigm option.
- [ ] Under that option, λ or platforms differ across paradigms *because of
      incentives*, not because of an optional rank-coupling flag.
- [ ] README states which theories the cell can and cannot distinguish.
- [ ] Default one-shot cell remains available and is labeled as a
      selection-rule comparison with exogenous types, not a test of loyalty
      formation.
- [ ] If strategic voting is added, sincere plurality remains an explicit
      baseline.

## Files

- `farm/experiments/consensus/` (new module, e.g. `mechanism.py`)
- `farm/experiments/consensus/paradigms.py`
- `farm/experiments/consensus/population.py`
- `farm/experiments/consensus/README.md`
- `docs/research/experiments/consensus_paradigms.md`
