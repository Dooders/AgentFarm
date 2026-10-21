# GCP Runbook: Cooperation in the Spatial World (seeds 101–120)

**Runs:** the registered experiment in [SpatialDesign.md](SpatialDesign.md) — four arms × seeds 101–120 × 3,000 steps, population cap 300
**Driver:** [`scripts/run_intrinsic_cooperation_experiment.py`](../../../../scripts/run_intrinsic_cooperation_experiment.py)
**Tooling added for this runbook:** [`scripts/gcp/intrinsic_cooperation_fleet.sh`](../../../../scripts/gcp/intrinsic_cooperation_fleet.sh), [`scripts/gcp/intrinsic_cooperation_startup.sh`](../../../../scripts/gcp/intrinsic_cooperation_startup.sh), [`scripts/merge_intrinsic_cooperation_runs.py`](../../../../scripts/merge_intrinsic_cooperation_runs.py)

The experiment is CPU-bound, memory-hungry and embarrassingly parallel across
seeds, so the recommended shape is **one Compute Engine VM per seed** (all
four arms run sequentially inside it), results synced to a Cloud Storage
bucket, and one merge step at the end that recomputes the registered
hypothesis family across the 20 seeds. A single-VM alternative is in
[§9](#9-alternative-one-vm-all-seeds).

Every command below runs from the repository root unless noted.

## 1. What a seed costs (measured)

These numbers come from local probes of the `baseline` arm on this branch
(4 vCPU, 16 GB host); the other arms carry the same brains plus the
cooperation service, so treat them as lower bounds and confirm them in
the pilot ([§5](#5-step-2-pilot-one-seed-at-full-parameters)).

| Quantity | Measured | Implication at the registered settings |
| --- | --- | --- |
| Resident memory per living agent | 45–68 MB (CNN DQN per agent; hidden size is heritable, up to 512) | 14–20 GB at the cap of 300 before any growth; budget **≥ 32 GB**, prefer 64 GB |
| Wall time per step | 0.36 s at 60 agents, 0.81 s at 180 agents (≈ 4.5–6 ms per agent-step) | ≈ 1.5–2 s/step at the cap → **1.3–1.7 h per arm**, **5–7 h per seed** (arms run one after another in the same process) |
| Population | reaches the cap within ~60 steps and stays there | the memory peak is reached early; a VM that survives step 100 at full population survives the run |
| Output per seed | ~20–60 MB (summary JSON, per-arm CSV/JSON, one PNG) | negligible storage cost |

The runner writes `intrinsic_cooperation_summary.json` **only when all its
replicates finish** and has **no resume**: an interrupted shard restarts from
step 0. That is why Spot VMs are not the default below.

Fleet cost (`n2-highmem-8`, 8 vCPU / 64 GB, us-central1 on-demand ≈ $0.47/h):
20 VMs × ~7 h ≈ **$65–70**; Spot ≈ $20–25 but with the restart risk above.
Storage, egress and the pilot add a few dollars.

## 2. Pre-flight gate

Do not launch the fleet until each of these holds. They are the open issues
from the methodology review of this branch; the fleet is cheap, but a result
that cannot be interpreted is not.

1. **Learning premise.** `S2` asks whether *the brain* learns to bond. In a
   400-step local probe the policies' softmax-over-Q stayed near uniform
   (mean max probability 0.17 vs 0.125 uniform) when the learning rate kept
   its default, so the action mix tracked the innate priors; with the
   platform's initial-diversity mutation the heritable learning rate drifted
   to a median of 0.2 and the Q-values became peaked but not meaningfully
   trained. Run the smoke test in §4 and compare `policy_bond_prob_late`
   between the cooperation and placebo arms; if they are indistinguishable at
   100 steps *and* at the pilot's 3,000 steps, the experiment measures
   selection on priors, not learning, and the design text should say so
   before the seeds are spent.
2. **Memory and time budget confirmed on the target machine type** by the
   pilot (§5): `resources.log` peak `mem_used_mb` < 80% of the VM, and
   `status.json` says `ok` with a wall time you are willing to pay 20× for.
3. **Population cap.** With the cap pinned from step ~60, `mean_population`
   (`S3`, `M2`) has almost no room to differ between arms. Either accept
   that `S3`/`M2` are tests of *time-to-cap* and *deaths*, or raise
   `--max-population` (memory scales with it — re-run the pilot) before
   registering the fleet's `EXTRA_ARGS`.
4. **Pinned commit.** The fleet runs whatever `GIT_REF` resolves to on the
   VMs. Push the branch first; the launcher refuses a HEAD that is not on
   any remote branch.

## 3. Prerequisites

- `gcloud` CLI installed and authenticated (`gcloud auth login`).
- A project with billing, Compute Engine and Cloud Storage enabled:

  ```bash
  gcloud config set project <PROJECT_ID>
  gcloud services enable compute.googleapis.com storage.googleapis.com
  ```

- A results bucket (regional, same region as the VMs keeps it free of
  egress):

  ```bash
  gcloud storage buckets create gs://<BUCKET> --location=us-central1 --uniform-bucket-level-access
  ```

- **Quota.** 20 × `n2-highmem-8` needs 160 N2 vCPUs in one region. New
  projects usually have 24–32. Check and request more, or launch in waves
  with `SEEDS` (§6):

  ```bash
  gcloud compute regions describe us-central1 --format="table(quotas.metric,quotas.usage,quotas.limit)" | grep -E "N2_CPUS|CPUS "
  ```

- The default Compute Engine service account must be able to write to the
  bucket. The launcher attaches the `storage-rw` scope; if you use
  uniform bucket-level access, also grant the role:

  ```bash
  PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')
  gcloud storage buckets add-iam-policy-binding gs://<BUCKET> \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" --role=roles/storage.objectAdmin
  ```

- Locally: a venv with the project installed (`pip install -r requirements.txt && pip install -e .`)
  for the merge step and the smoke test.

## 4. Step 1: smoke test (local, minutes)

Confirms the branch runs end to end and shows what one shard's output
looks like. It also gives a first reading on the learning premise.

```bash
source venv/bin/activate
PYTHONHASHSEED=0 python scripts/run_intrinsic_cooperation_experiment.py \
  --num-steps 100 --num-replicates 1 --seed 101 --max-population 60 \
  --output-dir /tmp/coop_smoke
python -c "import json; d=json.load(open('/tmp/coop_smoke/intrinsic_cooperation_summary.json'))['replicates'][0]; \
  print({a: (round(d[a]['mean_population'],1), round(d[a]['policy_bond_prob_late'],4)) for a in ('baseline','placebo','cooperation','cooperation_no_motive')})"
```

Expect (≈ 1.5 min on 4 vCPUs): four arm directories, `run_manifest.json`,
the summary and `intrinsic_cooperation_comparison.png`;
`aggregate`/`hypotheses` are `null` with one replicate. `PYTHONHASHSEED=0` is **not** set by this script (unlike
`run_simulation.py`), so set it yourself wherever you run it; the startup
script does.

## 5. Step 2: pilot, one seed at full parameters

One VM, seed 101, everything else registered. Keep the VM up afterwards so
you can inspect it.

```bash
export BUCKET=gs://<BUCKET> ZONE=us-central1-a
RUN_ID=coop-pilot SEEDS=101 KEEP_VM=true scripts/gcp/intrinsic_cooperation_fleet.sh launch
```

Watch it boot and install (≈ 5 min: apt, partial clone, CPU torch wheel,
requirements):

```bash
gcloud compute instances tail-serial-port-output coop-pilot-s101 --zone=$ZONE
```

While it runs (the shard syncs partial output every 10 minutes):

```bash
RUN_ID=coop-pilot BUCKET=$BUCKET scripts/gcp/intrinsic_cooperation_fleet.sh status
gcloud storage cat $BUCKET/intrinsic_cooperation/coop-pilot/seed_101/resources.log | tail -5   # time,load1,mem_used_mb,mem_available_mb,swap_used_mb
gcloud storage cat $BUCKET/intrinsic_cooperation/coop-pilot/seed_101/stderr.log | tail -20
```

or on the VM itself: `gcloud compute ssh coop-pilot-s101 --zone=$ZONE`, then
`sudo tail -f /var/log/intrinsic_cooperation_startup.log` and
`tail -f /opt/agentfarm/results/seed_101/stderr.log`.

Decide from the pilot:

| Observation | Action |
| --- | --- |
| peak `mem_used_mb` > 80% of RAM, or `swap_used_mb` > 0 | move to `MACHINE_TYPE=n2-highmem-16` (128 GB) or lower `--max-population` via `EXTRA_ARGS` and re-pilot |
| peak < 40% of RAM | `n2-highmem-4` (32 GB) is enough; halves the bill |
| wall time per seed | sets the fleet's expected duration; 20 VMs run in parallel so the fleet takes about as long as the pilot |
| `status.json` not `ok` | read `stderr.log`; fix on the branch, push, re-pilot with the new commit |

Keep the pilot's `host_manifest.json` (git commit, machine type, package
versions): it is the provenance record for the registered settings. Then
`RUN_ID=coop-pilot SEEDS=101 scripts/gcp/intrinsic_cooperation_fleet.sh delete`.

The pilot's seed 101 result is a valid shard of the final run **only if** it
ran the same commit and `EXTRA_ARGS` as the fleet; if so, copy its directory
into the fleet's prefix before merging, and launch the fleet with
`SEEDS="$(seq -s ' ' 102 120)"`.

## 6. Step 3: launch the fleet

```bash
export BUCKET=gs://<BUCKET> ZONE=us-central1-a
export RUN_ID=coop-$(git rev-parse --short HEAD)        # lowercase letters, digits, hyphens
export EXTRA_ARGS=""                                      # anything decided in §2/§5, e.g. "--max-population 200"
scripts/gcp/intrinsic_cooperation_fleet.sh launch         # SEEDS defaults to 101 ... 120
```

The launcher prints the resolved commit, machine type and the bucket prefix,
then creates one VM per seed named `<RUN_ID>-s<seed>`, labelled
`experiment=intrinsic-cooperation,run-id=<RUN_ID>`. Each VM:

1. installs the system packages, clones the repo at the pinned commit,
   builds a venv with the **CPU** torch wheel (the default wheel pulls CUDA
   and several GB for nothing), installs `requirements.txt` and the package;
2. writes `host_manifest.json` and `pip_freeze.txt`;
3. runs `scripts/run_intrinsic_cooperation_experiment.py --seed <seed> --num-replicates 1 --output-dir results/seed_<seed> $EXTRA_ARGS`
   with `PYTHONHASHSEED=0`, sampling memory and load every minute into
   `resources.log` and syncing partial output every 10 minutes;
4. writes `status.json` (`ok`/`failed`, exit code, timestamps), does a
   final sync to `gs://<BUCKET>/intrinsic_cooperation/<RUN_ID>/seed_<seed>/`,
   and shuts itself down (stopped VMs do not bill for CPU; disks do, until
   §10).

Quota-limited? Launch in waves with the same `RUN_ID`:

```bash
SEEDS="101 102 103 104" scripts/gcp/intrinsic_cooperation_fleet.sh launch
# ... when those finish:
SEEDS="105 106 107 108" scripts/gcp/intrinsic_cooperation_fleet.sh launch
```

Spot: `PROVISIONING=SPOT` cuts the price by ~65%. A preempted VM is
*stopped*, and its startup script re-runs the shard from scratch on the next
start (`gcloud compute instances start <name>`); you must restart it
yourself, and a shard that is preempted twice has cost more than on-demand.
Use it for exploratory runs, not the registered one.

## 7. Monitor

```bash
scripts/gcp/intrinsic_cooperation_fleet.sh status
```

prints the VMs (RUNNING → TERMINATED as shards finish) and every
`status.json` present in the bucket. Per-VM detail:

```bash
gcloud compute instances tail-serial-port-output ${RUN_ID}-s113 --zone=$ZONE
gcloud storage cat $BUCKET/intrinsic_cooperation/$RUN_ID/seed_113/resources.log | tail -3
```

A VM that is TERMINATED with no `status.json` in the bucket was stopped
before its exit trap ran (a manual stop, or the whole VM running out of
memory). Its `/var/log/intrinsic_cooperation_startup.log` is still on the
boot disk. Starting the VM re-runs the shard from scratch (the "done" marker
only exists after a successful run), so to inspect without re-running,
start it with the startup script disabled:

```bash
gcloud compute instances remove-metadata ${RUN_ID}-s113 --zone=$ZONE --keys=startup-script
gcloud compute instances start ${RUN_ID}-s113 --zone=$ZONE
```

## 8. Collect, merge, verify

```bash
scripts/gcp/intrinsic_cooperation_fleet.sh collect
```

downloads the whole prefix to `results/gcp/<RUN_ID>/`, lists any shard whose
`status.json` is not `ok`, and merges the good ones with
`scripts/merge_intrinsic_cooperation_runs.py` into
`results/gcp/<RUN_ID>/merged/`:

- `intrinsic_cooperation_summary.json` — the same shape the runner writes
  for `--num-replicates 20`: `config` (with `seeds`), `replicates`
  (re-indexed, sorted by seed), `aggregate` (per-arm descriptives and every
  seed-paired contrast), `hypotheses` (the nine-test Holm family S2, S3,
  S4×6, S5 followed by the uncorrected checks M1, M2), plus `sources`;
- `hypotheses.md` — that family as a table.

The merge refuses shards whose run configuration differs (other than seed,
replicate count and output paths) and duplicate seeds; it does **not**
redraw the comparison figure, which the runner only produces per process —
each shard's `intrinsic_cooperation_comparison.png` is still there. To merge
by hand (e.g. after copying the pilot's shard in):

```bash
python scripts/merge_intrinsic_cooperation_runs.py results/gcp/$RUN_ID/seed_*/intrinsic_cooperation_summary.json \
  --output-dir results/gcp/$RUN_ID/merged
```

Verification checklist before anything is written up:

- [ ] 20 shards with `status.json` → `ok`; `merged/.../config.seeds == [101, …, 120]`.
- [ ] every `host_manifest.json` has the same `git_commit`, `machine_type` and `extra_args`.
- [ ] every `run_manifest.json` shows the registered arguments (Appendix A of the design).
- [ ] no shard's `resources.log` shows `swap_used_mb > 0` (swapping changes nothing numerically but is the sign of a run at the edge of memory).
- [ ] `hypotheses` has 11 entries: 9 with `p_holm`, 2 flagged `manipulation_check`.
- [ ] a re-run of one seed (same commit, same VM type) reproduces its `replicates[0]` — `PYTHONHASHSEED=0` is set, so it should be bit-identical.

Then record the result in `RESULTS.md` next to [SpatialDesign.md](SpatialDesign.md),
update the catalog status in [experiments-catalog.md](../../experiments-catalog.md),
and commit `merged/intrinsic_cooperation_summary.json` and `hypotheses.md`
(not the per-arm CSVs) under `experiments/`.

## 9. Alternative: one VM, all seeds

Simplest when quota or budget rules out the fleet. One `n2-highmem-8`
running the runner's own replicate loop:

```bash
RUN_ID=coop-serial SEEDS=101 NUM_REPLICATES=20 KEEP_VM=false scripts/gcp/intrinsic_cooperation_fleet.sh launch
```

The runner then uses seeds 101–120 itself, computes the hypotheses in
process and writes the figure; no merge is needed (`collect` still works and
merges the single shard trivially). Expect **5–6 days** of wall time, during
which a single failure loses everything — the runner has no checkpoint —
so this path is only sensible after a clean pilot. Use the fleet's
`status`/`resources.log` to watch it exactly as in §7.

## 10. Teardown

```bash
scripts/gcp/intrinsic_cooperation_fleet.sh delete                           # the VMs (and boot disks) of this RUN_ID
gcloud compute instances list --filter="labels.experiment=intrinsic-cooperation"   # should be empty
```

Keep the bucket; add a lifecycle rule if you want the raw per-arm files
to expire:

```bash
echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":180}}]}' > /tmp/lifecycle.json
gcloud storage buckets update gs://<BUCKET> --lifecycle-file=/tmp/lifecycle.json
```

## 11. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| serial console stops after `pip install` lines, VM still RUNNING | The CPU torch wheel is ~200 MB and `requirements.txt` builds a few packages; allow 5–8 minutes. If longer, `ssh` in and read `/var/log/intrinsic_cooperation_startup.log`. |
| `status.json` → `failed`, `stderr.log` ends in `Killed` | OOM. Confirm with `resources.log` (memory rising to the VM size); use a larger machine type or lower `--max-population`. |
| `stderr.log`: `numpy` / `numba` import error | `requirements.txt` pins `numpy<2.3` for Tianshou's Numba; a stale venv on a reused VM can have a newer numpy. VMs are disposable — delete and relaunch. |
| step time much worse than the pilot's per-agent figure | Torch intra-op threads oversubscribed by hundreds of small per-agent networks. Relaunch with `OMP_THREADS=2` (or `4`); the value is recorded in `host_manifest.json`. Try it on the pilot VM first — it changes speed, not results. |
| two runs of the same seed differ | `PYTHONHASHSEED=0` missing (the CLI does not set it), different commit, or different torch wheel; compare `host_manifest.json` and `pip_freeze.txt`. |
| `permission denied` on `gcloud storage rsync` from the VM | Service account lacks bucket access (§3 IAM binding), or the VM was created without the `storage-rw` scope (launch again with the fleet script). |
| launcher: `HEAD … is not on any remote branch` | Push the branch, or set `GIT_REF` to a pushed commit/branch. |
| instance name rejected | `RUN_ID` must be lowercase letters, digits and hyphens, start with a letter, and keep `<RUN_ID>-s<seed>` under 63 characters. |
| VM reboots and starts the shard again | Intended for preempted Spot VMs; a finished shard is skipped via `/var/lib/intrinsic_cooperation/done`. |
