# On-chain run provenance (FarmNotary)

**Status:** Proposed  
**Sibling repo:** [Dooders/FarmNotary](https://github.com/Dooders/FarmNotary)

## Intent

AgentFarm executes simulations. FarmNotary attests what was published afterward.

The chain stores `manifest_hash + CID`. It does not run the simulator. Immutability is not correctness; re-execution from a committed seed is still required to test claims.

## Official record vs private choice

- **Private:** individual votes, agent-level ballots, raw choice vectors used only for selection.
- **Official (notarized):** config, git SHA, aggregate metrics, winner allocations, summary tables.

Filenames containing `ballot`, `vote`, `voter`, `individual_choice`, or `private` are skipped by FarmNotary.

## Inheritance

AgentFarm depends on FarmNotary as an optional extra, not a git submodule:

```text
pip install -e ".[notary]"
# or, until the extra is wired in pyproject:
pip install -e ../FarmNotary
```

Call site after a runner writes a run directory:

```python
from farm.provenance.notary import notarize_run_dir

receipt = notarize_run_dir(
    run_dir,
    runner="consensus_paradigms",
    config={"trials": 250, "population": "two_cluster"},
)
```

If `farm_notary` is not installed, `notarize_run_dir` writes nothing and returns `None`.

## Layout

| Path | Role |
|------|------|
| `farm/provenance/notary.py` | Optional adapter |
| `scripts/notarize_run.py` | CLI over a finished run dir |
| [Guide](../guides/farm-notary.md) | How to stamp a run |
| [Consensus paradigms](../research/experiments/consensus_paradigms.md) | First experiment that should use this |
