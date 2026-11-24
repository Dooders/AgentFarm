# GCP Runbook: Cooperation in the Spatial World (seeds 101–120)

**Runs:** the registered experiment in [SpatialDesign.md](SpatialDesign.md) — four arms × seeds 101–120 × 3,000 steps, population cap 300
**Driver:** [`scripts/run_intrinsic_cooperation_experiment.py`](../../../../scripts/run_intrinsic_cooperation_experiment.py)
**Tooling added for this runbook:** [`scripts/gcp/intrinsic_cooperation_fleet.sh`](../../../../scripts/gcp/intrinsic_cooperation_fleet.sh), [`scripts/gcp/intrinsic_cooperation_startup.sh`](../../../../scripts/gcp/intrinsic_cooperation_startup.sh), [`scripts/merge_intrinsic_cooperation_runs.py`](../../../../scripts/merge_intrinsic_cooperation_runs.py), [`scripts/calibrate_intrinsic_cooperation.py`](../../../../scripts/calibrate_intrinsic_cooperation.py)

The experiment is CPU-bound, memory-hungry and embarrassingly parallel across
seeds, so the recommended shape is **one Compute Engine VM per seed** (all
four arms run sequentially inside it), results synced to a Cloud Storage
bucket, and one merge step at the end that recomputes the registered
hypothesis family across the 20 seeds. A single-VM alternative is in
[§9](#9-alternative-one-vm-all-seeds).

Every command below runs from the repository root unless noted.

## 1. What a seed costs (measured)

These numbers come from the full-length local pilot (seeds 1–2, every
registered setting, commit `13ffe91e`, two seeds in parallel on a 22-core,
64 GB host; record in
[`experiments/intrinsic_cooperation_pilot/`](../../../../experiments/intrinsic_cooperation_pilot/calibration.md)).
An earlier estimate that assumed the population sits at the cap of 300
overstated memory and time about fourfold.

| Quantity | Measured | Implication |
| --- | --- | --- |
| Population | peaks at ~70 around step 100, then regulates itself at 30–50 (baseline/placebo ~33–35, cooperation arms ~45–50); the cap of 300 never binds | memory peaks in the first ~150 steps |
| Resident memory per seed process | ≈ 5 GB at peak (two processes added ≈ 10 GB to the host; no swap) | an 8 GB VM is tight; **16 GB is ample** |
| Wall time per arm | baseline ≈ 19 min, placebo ≈ 20 min, cooperation ≈ 26 min, `cooperation_no_motive` ≈ 30 min | **≈ 1.6 h per seed** (arms run one after another in the same process); the optional fifth arm adds ≈ 30 min |
| Output per seed | ≈ 12 MB (summary JSON, per-arm JSON/CSV/telemetry) | negligible storage cost |

The runner writes each arm's `arm_result.json` as it finishes and a
`replicate_summary.json` per seed; `intrinsic_cooperation_summary.json`
follows when every replicate is done. A restart with the same command line
**resumes at the granularity of a finished arm** (an interrupted arm is
re-run from its start). A finished arm is reused only if its stored
provenance (arm, seed, experiment config, base-config hash, commit,
dirty flag, hash seed) matches the current run; otherwise it is re-run, so
a reused output directory cannot leak results from another configuration. A pre-empted Spot VM is stopped, not deleted, and
the startup script re-runs the same command in the same `results/` directory
on the next start, so a pre-emption costs the interrupted arm (up to ~30 min)
plus the restart you have to issue by hand. The 10-minute bucket sync also
keeps the finished arms off the VM. Spot stays off by default because the
restart is manual and a shard pre-empted twice has cost more than on-demand.

Fleet cost (`n2-standard-4`, 4 vCPU / 16 GB, us-central1 on-demand ≈ $0.19/h):
20 VMs × ~2 h (install plus ~1.6 h run) ≈ **$8**; the scripts' default
`n2-highmem-8` also works at ≈ $20. **The fleet is optional:** one 64 GB
workstation with `--jobs 6` runs all 20 seeds in about 6 h
([§9](#9-alternative-one-vm-all-seeds)).
Storage, egress and the pilot add a few dollars.

## 2. Pre-flight gate

Do not launch the fleet until each of these holds. The first three are the
calibration gates of [SpatialDesign.md §8](SpatialDesign.md#8-calibration);
`scripts/calibrate_intrinsic_cooperation.py` evaluates them from a merged
summary. The fleet is cheap, but a result that cannot be interpreted is not.

1. **Learning premise (L1, L2).** `S2` asks whether *the brain* learns to
   bond. The design now pins Chromosomes A and C, turns the initial-diversity
   layer off and sharpens the policy softmax (temperature 0.1); the reduced
   local pilot in §8 of the design passes both positive controls under that
   regime. The full-length pilot (§5) must pass them too: run the
   calibration report on its summary merged with at least one more seed, or
   read `policy_gather_lift_late` and `gather_share_over_prior_late` off its
   `replicate_summary.json` for the baseline arm — both must be positive. If
   they are not, do not launch; the experiment would measure selection on
   priors, not learning.
2. **Regime.** The same report's `units_form`, `generations`, `not_pinned`
   and `exit_cost_bounded` gates must pass on the full-length pilot. If
   `not_pinned` fails (an arm's mean population ≥ 95% of the cap), raise
   `--max-population` (memory scales with it — re-run the pilot) or raise
   `--selection-pressure` to `high` before registering the fleet's
   `EXTRA_ARGS`.
3. **Memory and time budget confirmed on the target machine type** by the
   pilot (§5): `resources.log` peak `mem_used_mb` < 80% of the VM, and
   `status.json` says `ok` with a wall time you are willing to pay 20× for.
4. **Pinned commit.** The fleet runs whatever `GIT_REF` resolves to on the
   VMs. Push the branch first; the launcher refuses a HEAD that is not on
   any remote branch. Every summary and `run_manifest.json` records the
   commit it ran from, and the merge refuses shards whose configuration
   (including the commit) differs.

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
`aggregate`/`hypotheses` are `null` with one replicate. The script restarts
itself with `PYTHONHASHSEED=0` when it is not already set (as
`run_simulation.py` does); the summary records `python_hash_seed`,
`git_commit` and `git_dirty`, and the merge refuses shards that differ in
any of them.

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
| peak < 40% of RAM | a smaller machine is enough; the local pilot peaked at ≈ 5 GB per seed, so `n2-standard-4` (16 GB) fits |
| wall time per seed | sets the fleet's expected duration; 20 VMs run in parallel so the fleet takes about as long as the pilot |
| `status.json` not `ok` | read `stderr.log`; fix on the branch, push, re-pilot with the new commit |
| a calibration gate fails (below) | do not launch; see §2 |

Run the calibration gates on the pilot. The report needs two or more
replicates for its paired statistics, so pilot two seeds (`SEEDS=101,102`)
or merge the pilot with a second shard; with a single seed, read the two
learning numbers directly:

```bash
RUN_ID=coop-pilot BUCKET=$BUCKET RESULTS_DIR=results/pilot scripts/gcp/intrinsic_cooperation_fleet.sh collect
python scripts/merge_intrinsic_cooperation_runs.py "results/pilot/seed_*/intrinsic_cooperation_summary.json" --output-dir results/pilot/merged
python scripts/calibrate_intrinsic_cooperation.py results/pilot/merged/intrinsic_cooperation_summary.json
# single seed:
python -c "import json; b=json.load(open('results/pilot/seed_101/replicate_summary.json'))['baseline']; \
  print('L1', b['policy_gather_lift_late'], 'L2', b['gather_share_over_prior_late'])"
```

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

The registered regime (temperature 0.1, frozen Chromosomes A/C, selection
`medium`, exit cost on the received base, masking, partner protection) is the
CLI's default, so `EXTRA_ARGS` stays empty unless the pilot forced a change.
To add the optional fifth arm, set
`EXTRA_ARGS="--arms baseline placebo cooperation cooperation_no_motive cooperation_zero_synergy"`
(about 25% more wall time per seed; every shard must use the same list or the
merge refuses them).

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
*stopped*; on the next start (`gcloud compute instances start <name>`) its
startup script re-runs the shard in the same `results/` directory, so the
runner loads the arms that finished and re-runs only the interrupted one.
You must restart it yourself, and a shard that is preempted twice has cost
more than on-demand. Use it for exploratory runs, not the registered one.

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
  seed-paired contrast), `hypotheses` (the seven-test Holm family S2, S3,
  S3b, S4×3, S5 followed by the uncorrected manipulation checks M1, M2, the
  learning checks L1, L2 and the exploratory contrasts, each flagged), plus
  `sources`;
- `hypotheses.md` — that table, with every out-of-family entry labelled.

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
- [ ] `hypotheses` has 7 entries with `p_holm` (S2, S3, S3b, S4_scope, S4_exit_cost, S4_recognition, S5); the rest are flagged `manipulation_check`, `learning_check` or `exploratory` and have `p_holm: null`.
- [ ] L1 and L2 are positive in the merged run, not just in the pilot (`scripts/calibrate_intrinsic_cooperation.py merged/intrinsic_cooperation_summary.json`).
- [ ] a re-run of one seed (same commit, same VM type) reproduces its `replicates[0]` — `PYTHONHASHSEED=0` is set, so it should be bit-identical.

Then record the result in `RESULTS.md` next to [SpatialDesign.md](SpatialDesign.md),
update the catalog status in [experiments-catalog.md](../../experiments-catalog.md),
and commit `merged/intrinsic_cooperation_summary.json` and `hypotheses.md`
(not the per-arm CSVs) under `experiments/`.

## 9. Alternative: one VM, all seeds

Simplest overall, and the default now that a seed costs ~5 GB and ~1.6 h.
On a workstation with 64 GB and at least 6 free cores:

```bash
python scripts/run_intrinsic_cooperation_experiment.py --num-steps 3000 --seed 101 \
  --num-replicates 20 --jobs 6 --output-dir experiments/intrinsic_cooperation_confirmatory
```

or on one VM (`n2-standard-8` or `n2-highmem-8`):

```bash
RUN_ID=coop-serial SEEDS=101 NUM_REPLICATES=20 EXTRA_ARGS="--jobs 6" KEEP_VM=false \
  scripts/gcp/intrinsic_cooperation_fleet.sh launch
```

The runner uses seeds 101–120 itself, computes the hypotheses in process and
writes the figure; no merge is needed. Expect about **6 h** of wall time
(20 seeds × 1.6 h / 6 jobs). A failure loses only the arms that were in
progress: re-running the same command reloads every finished arm whose
provenance matches. Commit before launching so the summary records a clean
tree.

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
| two runs of the same seed differ | different commit, a dirty tree (`git_dirty: true` in the summary), or a different torch wheel; compare `host_manifest.json` and `pip_freeze.txt`. |
| `permission denied` on `gcloud storage rsync` from the VM | Service account lacks bucket access (§3 IAM binding), or the VM was created without the `storage-rw` scope (launch again with the fleet script). |
| launcher: `HEAD … is not on any remote branch` | Push the branch, or set `GIT_REF` to a pushed commit/branch. |
| instance name rejected | `RUN_ID` must be lowercase letters, digits and hyphens, start with a letter, and keep `<RUN_ID>-s<seed>` under 63 characters. |
| VM reboots and starts the shard again | Intended for preempted Spot VMs; a finished shard is skipped via `/var/lib/intrinsic_cooperation/done`. |
