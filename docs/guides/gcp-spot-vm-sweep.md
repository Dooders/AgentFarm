## Running a sweep on a GCP Spot VM (startup and teardown)

This runbook covers spinning up a cheap, throwaway Google Cloud VM to run a
batch sweep (e.g. the intrinsic-goals selection-pressure sweep), pulling the
results back, and **deleting the VM so it stops billing**. It is written for a
one-off CPU batch job: many tiny, independent, single-threaded simulations with
no GPU.

The concrete values below (`agent-farm-experiments`, `us-central1-a`,
`agentfarm-sweep`) are the ones this project has used; substitute your own
project/zone/name as needed.

### Cost / shape at a glance

- `n2-standard-8` Spot in `us-central1` is roughly **$0.10/hr**, so a full
  sweep is well under $1.
- A **stopped** VM still bills for its boot disk, which is why teardown
  (deleting, not just stopping) matters.
- Each simulation is effectively single-threaded (many tiny DQNs), so pin BLAS
  threads to 1 and parallelize across *processes* (one per pressure level).

### 0. One-time local setup

```bash
gcloud auth login
gcloud config set project agent-farm-experiments
gcloud services enable compute.googleapis.com
gcloud config set compute/zone us-central1-a
```

If `services enable` fails with a billing error, link a billing account under
**Billing -> Link a billing account** in the console, then re-run.

### 1. Start the VM (Spot, no GPU)

The repo targets Python 3.12, so use an Ubuntu 24.04 LTS image (its default
`python3` is 3.12).

```bash
gcloud compute instances create agentfarm-sweep \
  --machine-type=n2-standard-8 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --boot-disk-type=pd-balanced \
  --metadata=enable-guest-attributes=TRUE
```

`enable-guest-attributes=TRUE` is required so the matrix orchestrator can publish
live progress to guest attributes (readable **without SSH**).

Verify it is up:

```bash
gcloud compute instances list --format='table(name,zone,status,machineType)'
```

> **Connectivity gotcha:** plain `gcloud compute ssh`/`scp` may hang or time out
> on this project (no public-ingress SSH path). If that happens, add
> `--tunnel-through-iap` to **every** `ssh`/`scp` command. Examples below show
> the IAP form. IAP tunneling requires the caller to have the
> `roles/iap.tunnelResourceAccessor` role.

### 2. Provision the environment on the VM

```bash
gcloud compute ssh agentfarm-sweep --zone=us-central1-a --tunnel-through-iap --command='
set -e
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git python3.12-venv python3-tk
[ -d AgentFarm ] || git clone --depth 1 https://github.com/Dooders/AgentFarm.git
cd AgentFarm
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -e .
python -c "import farm; print(\"farm ok\", farm.__file__)"
echo SETUP_COMPLETE
'
```

### 3. Push local-only scripts (if not yet on GitHub)

The VM cloned `main` from GitHub. Any changes that only exist locally (for
example new/edited sweep or runner files) must be copied over before running:

```bash
gcloud compute scp --tunnel-through-iap \
  scripts/run_intrinsic_goals_pressure_sweep.py \
  scripts/analyze_intrinsic_goals_pressure_sweep.py \
  farm/runners/intrinsic_goals_experiment.py \
  agentfarm-sweep:~/AgentFarm/ --zone=us-central1-a
```

Adjust the file list to whatever you changed. Mirror the repo layout on the
destination (scripts under `~/AgentFarm/scripts/`, package files under their
package path). When in doubt, `scp` the specific edited files into their exact
paths.

### 4. Launch the sweep (detached, survives SSH drops)

Run the three pressure levels as parallel processes so the wall time is set by
the slowest single level, not the sum. Use a small launch script uploaded to
the VM to avoid SSH quoting pitfalls.

Create `start_sweep.sh` locally:

```bash
cat > /tmp/start_sweep.sh << 'EOF'
#!/bin/bash
set -euo pipefail
cd ~/AgentFarm
source venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p experiments
echo "[$(date -Iseconds)] starting 3-pressure sweep" | tee experiments/sweep_master.log
for sp in low medium high; do
  python scripts/run_intrinsic_goals_experiment.py \
    --num-steps 600 --seed 42 --num-replicates 20 \
    --selection-pressure "$sp" --max-population 3000 \
    --output-dir "experiments/intrinsic_goals_sweep_${sp}" --log-level WARNING \
    > "experiments/sweep_${sp}.log" 2>&1 &
  echo "[$(date -Iseconds)] launched $sp pid $!" | tee -a experiments/sweep_master.log
done
wait
echo "[$(date -Iseconds)] all pressures done; analyzing" | tee -a experiments/sweep_master.log
python scripts/analyze_intrinsic_goals_pressure_sweep.py --sweep-dir experiments \
  >> experiments/sweep_master.log 2>&1
echo "[$(date -Iseconds)] DONE" | tee -a experiments/sweep_master.log
EOF
```

Upload and start it detached:

```bash
gcloud compute scp --tunnel-through-iap /tmp/start_sweep.sh \
  agentfarm-sweep:~/start_sweep.sh --zone=us-central1-a

gcloud compute ssh agentfarm-sweep --zone=us-central1-a --tunnel-through-iap --command='
chmod +x ~/start_sweep.sh
nohup ~/start_sweep.sh >> ~/AgentFarm/experiments/nohup.out 2>&1 </dev/null & disown
sleep 3
pgrep -af "start_sweep|run_intrinsic_goals" || echo NO_PROCS
cat ~/AgentFarm/experiments/sweep_master.log 2>/dev/null || echo no_master_yet
'
```

Each pressure level now runs 3 arms (`uniform`, `shared`, `unique`) x 20
replicates x 600 steps. Budget roughly **3-5 hours** wall time for the full job;
`analyze_...` runs automatically at the end.

### 5. Monitor progress (prefer no SSH)

**Why SSH keeps dying:** packing the VM with one worker per vCPU starves
`sshd` / the guest agent. Under load, IAP can hang at the SSH banner even while
the instance stays `RUNNING` and CPU looks only partially used. Status checks
must not depend on an interactive shell.

**Preferred — guest attributes (no SSH):**

```bash
# One-shot / watch loop from your laptop
python scripts/check_gcp_matrix_status.py
python scripts/check_gcp_matrix_status.py --watch 60
```

Or raw `gcloud`:

```bash
gcloud compute instances get-guest-attributes agentfarm-sweep \
  --zone=us-central1-a \
  --query-path=status/matrix \
  --format=get(value)
```

The matrix orchestrator (`scripts/run_intrinsic_evolution_matrix.py`) writes
`matrix_live_status.json` under the output dir and mirrors a compact copy to
guest attribute `status/matrix` on a heartbeat (default 60s) and after each
job completes.

**SSH headroom when launching the matrix:** leave a core free and nice workers
(defaults already do this: `--jobs` = `nproc - 1`, children `nice +10`):

```bash
# On the VM (example)
python scripts/run_intrinsic_evolution_matrix.py \
  --jobs $(( $(nproc) - 1 )) \
  --disk-database --resume \
  --output-dir experiments/intrinsic_matrix
```

**Legacy SSH peek** (may time out under load — use only as fallback):

```bash
gcloud compute ssh agentfarm-sweep --zone=us-central1-a --tunnel-through-iap \
  --ssh-flag='-o ConnectTimeout=15' --command='
tail -3 ~/AgentFarm/experiments/sweep_master.log
pgrep -c -f run_intrinsic_goals_experiment.py
ls ~/AgentFarm/experiments/intrinsic_goals_sweep_*/intrinsic_goals_summary.json 2>/dev/null || echo summaries_not_ready
'
```

For older intrinsic-goals sweeps, the job is finished when `DONE` appears in
`sweep_master.log` and `combined_comparison.md` /
`intrinsic_goals_pressure_sweep.png` exist under `experiments/`. For the
population matrix, finish when `check_gcp_matrix_status.py` shows
`n_pending: 0` and `note: finished`.

### 6. Pull results back to your local machine

```bash
gcloud compute scp --recurse --tunnel-through-iap \
  agentfarm-sweep:~/AgentFarm/experiments ./gcp-results --zone=us-central1-a
```

### 7. Teardown (do this, or you keep paying)

Delete the VM (removes the instance *and* its boot disk):

```bash
gcloud compute instances delete agentfarm-sweep --zone=us-central1-a --quiet
```

Confirm nothing is left running/billing:

```bash
gcloud compute instances list
gcloud compute disks list
```

### Gotchas

- **SSH under load is unreliable:** do not use SSH as the primary status channel.
  Enable guest attributes at create time, leave one vCPU free (`--jobs nproc-1`),
  and poll with `scripts/check_gcp_matrix_status.py`. If you forgot guest
  attributes on an existing VM:
  `gcloud compute instances add-metadata agentfarm-sweep --metadata=enable-guest-attributes=TRUE`
  (then restart the matrix process so it can publish).
- **IAP tunnel:** if `ssh`/`scp` hangs, use `--tunnel-through-iap` on every
  command (see step 1). Add connect timeouts so a flaky tunnel fails fast
  instead of hanging: `--ssh-flag='-o ConnectTimeout=15'` for `ssh`, and
  `--scp-flag='-o ConnectTimeout=15'` for `scp` (`scp` does **not** accept
  `--ssh-flag`; it errors with a usage message).
- **Duplicate launches:** a launch attempt that *looks* hung (tunnel stalled
  before echoing output) may still have started the sweep on the VM. Before
  retrying, count runners and kill any strays:

  ```bash
  gcloud compute ssh agentfarm-sweep --zone=us-central1-a --tunnel-through-iap \
    --command='pgrep -cf "[r]un_intrinsic_goals_experiment.py"'
  ```

  Expect exactly one process per pressure level. If you must kill and restart,
  also `rm -rf ~/AgentFarm/experiments` so partial artifacts from the aborted
  run don't mix with the fresh one.
- **`pkill` kills your own SSH session:** the remote command line itself
  contains the pattern, so `pkill -f start_sweep.sh` over SSH matches (and
  kills) the shell running it, and the connection dies with exit 255. Use a
  self-excluding bracket pattern: `pkill -f "[s]tart_sweep.sh"`.
- **Wall time scales with the population cap:** with `--max-population 3000`
  a 600-step sim runs ~1.5-1.9 s/step (~15 min/sim), so 60 sims per pressure
  (20 replicates x 3 arms) is roughly **12-16 hours**, not the 3-5 hours the
  small-cap pilot suggested. Cost is still only ~$1-2 on Spot.
- **Spot preemption:** if the VM is preempted mid-run it STOPs. Restart with
  `gcloud compute instances start agentfarm-sweep --zone=us-central1-a`,
  re-SSH, and re-run step 4 — at most the interrupted pressure level is redone.
- **Stopped != free:** a stopped VM still bills for its disk. Always delete
  (step 7) when done rather than leaving it stopped.
- **Thread pinning:** always export `OMP_NUM_THREADS=1` (and the MKL/OpenBLAS/
  NumExpr equivalents). Oversubscribed BLAS threads slow these many-tiny-model
  runs down.
- **Fully hands-off variant:** attach a
  `--metadata-from-file startup-script=...` that clones, installs, runs the
  sweep, uploads `experiments/` to a GCS bucket, then `shutdown -h now`, so you
  never pay for idle. Harder to debug on the first run, cheapest thereafter.
