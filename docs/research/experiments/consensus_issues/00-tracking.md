---
title: "Consensus experiment: remaining design flaws that leave the headline question unanswerable"
type: Task
labels: [Experiment]
---

## Context

The political consensus experiment (`farm/experiments/consensus/`, CLI
`run_experiment.py`) asks whether individual-centered selection produces
stewards who treat electoral non-supporters better than party selection.

A design critique of the original spec is still largely applicable. Two items
were repaired and should **not** be re-litigated:

- Prefs and benefits are now linked: `benefits = softmax(prefs, T=0.55)` in
  `farm/experiments/consensus/population.py`.
- Benefits have real cluster structure, so `dir_supporters` and `dir_all`
  diverge as `N` grows (cluster-mean L1 distance stays ~0.85–0.92 from N=40 to
  N=4000). The prototype's "law of large numbers kills the treatment" finding
  no longer holds for this package.

Trials are also already paired: every paradigm shares the population and
candidate slate within a trial.

What remains is that the **headline hypothesis is still unanswerable in the
default condition**, welfare comparisons still mix incomparable estimands, and
the report still narrates results as if they tested loyalty selection.

Default committed run (`results/consensus/REPORT.md`, two_cluster, 250×400×8):

| Paradigm | λ_winner | loser_share | loser_welfare | total_welfare |
|---|---|---|---|---|
| party | 0.51 | 0.50 | 0.158 | 0.233 |
| individual | 0.51 | 0.70 | 0.197 | 0.232 |
| score | 0.51 | 0.82 | 0.217 | 0.230 |
| latent_match | 0.51 | 0.84 | 0.219 | 0.229 |

`λ_winner` is flat by construction. Loser-welfare "gains" track the changing
loser set, not loyalty.

A second, older scaffold (`farm/runners/consensus_paradigms_experiment.py` +
`docs/research/experiments/consensus_paradigms.md`) still matches the original
prototype more closely.

## Goal

Make the experiment able to answer a well-posed question, then regenerate
reports from that design. Child issues cover the remaining fatal / serious /
process items.

## Already fixed (do not redo)

- [x] Map `prefs → benefits` (softmax)
- [x] Cluster-structured benefits so allocation directions can diverge
- [x] Shared population + candidate draw across paradigms within a trial
- [x] README labels `constrained_individual` as a constitutional contrast, not
      a voting rule (still reported as a peer row when enabled)

## Open work

1. Make λ observable to voters **or** drop the λ-selection hypothesis;
   pre-register the primary condition (`--lambda-correlated` currently flips
   the core result).
2. Normalize the winner platform before the 0.72/0.28 blend (clip-sums
   currently range ~1.1–2.8, so "28% platform" is 30–52% after normalization).
3. Report welfare on a **fixed** partition (cluster A vs B), plus random-winner
   / utilitarian brackets and distributional stats (min, p10, Gini).
4. Infer from the paired design: CIs, paired tests, effect sizes, multiplicity
   correction.
5. Make project names match mechanics, or strip them; persist synthetic
   ballots / supporter masks for audit.
6. Replace tautological invariants with property tests (monotonicity,
   relabeling, λ=0 maximizes total welfare in-family).
7. Analysis hygiene: do not treat `constrained_individual` as a peer rule;
   delete prototype-orientation numbers from the scaffold spec; retire or
   rebase the scaffold runner.
8. Mechanism (larger): endogenous λ via entry / platforming / re-election;
   strategic voting. Without this the design cannot adjudicate
   citizen-candidate vs core/swing-voter predictions.

## Acceptance

- [ ] Each child issue closed or explicitly deferred with a written reason in
      `farm/experiments/consensus/README.md`.
- [ ] Default run no longer reports "λ_winner unchanged" as if it were an
      empirical finding while λ is independent of anything voters see.
- [ ] `REPORT.md` evaluates a question the generator can actually answer.

## Related

- Package: `farm/experiments/consensus/`
- Scaffold leftover: `farm/runners/consensus_paradigms_experiment.py`,
  `docs/research/experiments/consensus_paradigms.md`
- Notary policy (do not notarize voter-level choices): #983 — audit
  persistence can stay off the official record
