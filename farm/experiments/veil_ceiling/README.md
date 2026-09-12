# `farm.experiments.veil_ceiling`

Self-contained implementation of the pre-registered experiment *The Veil
Ceiling: Observation Collapse in Sealed-World Agent Evaluation*. Design,
Appendix A parameters and deviations: [docs/research/experiments/veil_ceiling/Design.md](../../../docs/research/experiments/veil_ceiling/Design.md).
Results: [RESULTS.md](../../../docs/research/experiments/veil_ceiling/RESULTS.md).

```bash
PYTHONHASHSEED=0 python scripts/run_veil_ceiling.py                 # full 780-run matrix
python scripts/run_veil_ceiling.py --analyze-only                    # re-analyse committed raw outputs
python scripts/run_veil_ceiling.py --seeds 3 --no-robustness --output-dir /tmp/veil_pilot
python scripts/run_veil_ceiling_validity_followup.py            # feature ablation + honest-calibrated evaluator
```

| Module | Responsibility |
|---|---|
| `config.py` | Pre-registered constants, `WorldConfig`, `MonitoringConfig`, `LearnerConfig`, `Condition`, `CONDITIONS`, `RunConfig`, `AnalysisThresholds` |
| `world.py` | `ResourceField`: logistic regeneration, gather vs over-harvest, defection opportunities |
| `monitoring.py` | Epoch-resampled true/decoy monitor masks and the binary-symmetric cue channel |
| `learner.py` | `MLPQLearner`: numpy Q-learner exposing `get_model_state` / `load_model_state` / `policy` for Lamarckian warm-start |
| `agents.py` | `VeilAgent` and the phase × region × monitored × cue `AgentLedger` |
| `simulation.py` | `VeilSimulation`: tick loop, penalties, reproduction, warm-start via `farm.core.policy_inheritance` |
| `metrics.py` | Divergence, predictive validity (LOSO AUC), strategy classes, onset, held-out transfer |
| `experiment.py` | `MatrixConfig`, `run_matrix`, `load_outputs`: cells × seeds × inheritance modes with raw outputs on disk |
| `analysis.py` | Seed-paired contrasts, C4 noise band, dose-response, falsification checks, hypothesis verdicts |
| `report.py` | `REPORT.md` and figures |
| `validity_followup.py` | Exploratory follow-up to H2: feature ablation, C1/C4-calibrated evaluator, leak profiles (`VALIDITY_FOLLOWUP.md`) |
