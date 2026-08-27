"""Consensus-paradigms experiment runner (scaffold).

Compares party / individual / score / latent_match selection, then a steward
allocates a budget. Optional FarmNotary stamp of official artifacts only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from farm.provenance.notary import notarize_run_dir

PARADIGMS = ("party", "individual", "score", "latent_match", "constrained_individual")
PROJECTS = (
    "core_services",
    "coalition_club",
    "outgroup_repair",
    "prestige_project",
    "buffer_reserve",
)


@dataclass
class TrialRow:
    paradigm: str
    seed: int
    winner: int
    total_welfare: float
    supporter_welfare: float
    loser_welfare: float
    lambda_winner: float
    loser_share: float


def _population(n_voters: int, n_cand: int, rng: np.random.Generator):
    n1 = n_voters // 2
    cluster_a = rng.normal([-0.8, 0.4, -0.5, 0.2, 0.1], 0.55, size=(n1, 5))
    cluster_b = rng.normal([0.8, -0.3, 0.6, 0.1, 0.0], 0.55, size=(n_voters - n1, 5))
    prefs = np.vstack([cluster_a, cluster_b])
    benefits = np.clip(prefs, 0.05, None)
    benefits = benefits / benefits.sum(axis=1, keepdims=True)
    cplat = rng.normal(0.0, 0.7, size=(n_cand, 5))
    loyalty = rng.beta(2.2, 2.2, size=n_cand)
    pplat = np.stack([cluster_a.mean(0), cluster_b.mean(0)])
    party_id = np.argmin(
        np.linalg.norm(cplat[:, None, :] - pplat[None, :, :], axis=2), axis=1
    )
    return prefs, benefits, cplat, loyalty, party_id, pplat


def _nearest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).argmin(axis=1)


def _allocate(winner_plat, winner_loy, benefits, supporter_mask):
    supp = benefits[supporter_mask]
    dir_s = supp.mean(0) if len(supp) else benefits.mean(0)
    dir_all = benefits.mean(0)
    raw = winner_loy * dir_s + (1 - winner_loy) * dir_all
    raw = 0.72 * raw + 0.28 * np.clip(winner_plat, 0, None)
    raw = np.clip(raw, 1e-6, None)
    return raw / raw.sum()


def run_once(paradigm: str, n_voters: int, n_cand: int, seed: int, lambda_cap: float = 0.25) -> TrialRow:
    rng = np.random.default_rng(seed)
    prefs, benefits, cplat, loyalty, party_id, pplat = _population(n_voters, n_cand, rng)

    if paradigm == "party":
        vote_party = _nearest(prefs, pplat)
        nominees = []
        for p in (0, 1):
            members = np.where(party_id == p)[0]
            if len(members) == 0:
                members = np.array([p % n_cand])
            nominees.append(
                int(members[np.argmin(np.linalg.norm(cplat[members] - pplat[p], axis=1))])
            )
        tally = np.bincount(vote_party, minlength=2)
        win_party = int(tally.argmax())
        winner = nominees[win_party]
        support = vote_party == win_party
        used_lambda = float(loyalty[winner])
    elif paradigm in ("individual", "constrained_individual"):
        choice = _nearest(prefs, cplat)
        winner = int(np.bincount(choice, minlength=n_cand).argmax())
        support = choice == winner
        used_lambda = float(loyalty[winner])
        if paradigm == "constrained_individual":
            used_lambda = min(used_lambda, lambda_cap)
    elif paradigm == "score":
        dist = np.linalg.norm(prefs[:, None, :] - cplat[None, :, :], axis=2)
        scores = 10 * (1 - dist / (dist.max() + 1e-9))
        winner = int(scores.mean(0).argmax())
        support = scores.argmax(1) == winner
        used_lambda = float(loyalty[winner])
    elif paradigm == "latent_match":
        winner = int(np.linalg.norm(cplat - prefs.mean(0), axis=1).argmin())
        choice = _nearest(prefs, cplat)
        support = choice == winner
        used_lambda = float(loyalty[winner])
    else:
        raise ValueError(paradigm)

    alloc = _allocate(cplat[winner], used_lambda, benefits, support)
    u = benefits @ alloc
    losers = ~support
    return TrialRow(
        paradigm=paradigm,
        seed=seed,
        winner=winner,
        total_welfare=float(u.mean()),
        supporter_welfare=float(u[support].mean()) if support.any() else float("nan"),
        loser_welfare=float(u[losers].mean()) if losers.any() else float("nan"),
        lambda_winner=float(loyalty[winner]),
        loser_share=float(losers.mean()),
    )


class ConsensusParadigmsExperiment:
    def __init__(self, output_dir: str | Path = "experiments/consensus_paradigms"):
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        trials: int = 50,
        voters: int = 200,
        candidates: int = 8,
        paradigms: Optional[Iterable[str]] = None,
        notarize: bool = True,
    ) -> Path:
        paradigms = tuple(paradigms or PARADIGMS)
        rows: list[TrialRow] = []
        for paradigm in paradigms:
            for seed in range(trials):
                rows.append(run_once(paradigm, voters, candidates, seed))

        trials_path = self.results_dir / "trials.csv"
        header = ",".join(TrialRow.__dataclass_fields__)
        body = "\n".join(",".join(str(getattr(r, k)) for k in TrialRow.__dataclass_fields__) for r in rows)
        trials_path.write_text(header + "\n" + body + "\n", encoding="utf-8")

        summary = []
        for paradigm in paradigms:
            subset = [r for r in rows if r.paradigm == paradigm]
            def mean(attr: str) -> float:
                return float(np.nanmean([getattr(r, attr) for r in subset]))

            summary.append({
                "paradigm": paradigm,
                "total_welfare": mean("total_welfare"),
                "supporter_welfare": mean("supporter_welfare"),
                "loser_welfare": mean("loser_welfare"),
                "gap": mean("supporter_welfare") - mean("loser_welfare"),
                "lambda_winner": mean("lambda_winner"),
                "loser_share": mean("loser_share"),
            })
        summary_path = self.results_dir / "summary.csv"
        keys = list(summary[0].keys())
        summary_path.write_text(
            ",".join(keys) + "\n" + "\n".join(",".join(str(s[k]) for k in keys) for s in summary) + "\n",
            encoding="utf-8",
        )
        meta = {"trials": trials, "voters": voters, "candidates": candidates, "paradigms": list(paradigms)}
        (self.results_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        if notarize:
            notarize_run_dir(
                self.results_dir,
                runner="consensus_paradigms",
                config=meta,
                official_record={"summary": summary},
            )
        return summary_path
