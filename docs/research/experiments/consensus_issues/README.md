# Consensus experiment — issue drafts

Ready-to-file GitHub issues for the remaining design flaws in the political
consensus experiment (`farm/experiments/consensus/`). These are drafts, not
live tickets: the environment that produced them could read
`Dooders/AgentFarm` issues but could not create them (`403 Resource not
accessible by personal access token`).

File them (once a token with `issues:write` is available):

```bash
# from repo root
for f in docs/research/experiments/consensus_issues/0*.md; do
  title=$(python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]).read().split('---',2)[1])['title'])" "$f")
  gh issue create --repo Dooders/AgentFarm --title "$title" --body-file "$f"
done
```

`gh` will include the YAML front matter in the body. Strip it first if you
want a cleaner issue, or paste each file's body after the second `---` by hand.

## Already fixed — do not re-file

- Prefs map to benefits: `benefits = softmax(prefs, T=0.55)`.
- Benefits have cluster structure, so `dir_supporters` and `dir_all` diverge
  as `N` grows (the original "law of large numbers kills the treatment"
  finding does not hold for this package).
- Paradigms share the population and candidate slate within a trial.
- README already labels `constrained_individual` as a constitutional contrast
  (it is still rendered as a peer row when enabled — see `07`).

## Drafts

| File | Suggested type | Suggested labels | What it fixes |
|---|---|---|---|
| [00-tracking.md](00-tracking.md) | Task | Experiment | Parent / remaining-work map |
| [01-lambda-unobservable.md](01-lambda-unobservable.md) | Bug | Experiment | Headline λ hypothesis is impossible by default |
| [02-normalize-platform-blend.md](02-normalize-platform-blend.md) | Bug | Experiment | 0.72/0.28 mix is not a real blend |
| [03-fixed-partition-welfare.md](03-fixed-partition-welfare.md) | Feature | Experiment | Incomparable supporter estimands, no baselines, means-only, `rural_town` size confound |
| [04-paired-inference.md](04-paired-inference.md) | Feature | Experiment | No CIs, paired tests, effect sizes, or multiplicity correction |
| [05-project-names-and-ballots.md](05-project-names-and-ballots.md) | Bug | Experiment | Fictional project semantics; ballots not auditable |
| [06-property-tests.md](06-property-tests.md) | Task | Experiment | Invariants test assumptions, not correctness |
| [07-analysis-hygiene.md](07-analysis-hygiene.md) | Task | Experiment | `constrained_individual` as peer; prototype-orientation numbers; leftover scaffold |
| [08-endogenous-loyalty.md](08-endogenous-loyalty.md) | Feature | Experiment | No mechanism for the thing being studied; all voters sincere |

Related: scaffold leftover `#983` (FarmNotary / do-not-notarize voter choices).
Audit persistence in `05` can stay off the official notary record.
