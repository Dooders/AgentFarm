#!/usr/bin/env bash
# Compute Engine startup script: run ONE shard of the intrinsic cooperation
# experiment on a fresh Ubuntu 24.04 VM and copy the results to Cloud Storage.
#
# Configured through instance metadata (see intrinsic_cooperation_fleet.sh,
# which sets these when it creates the VMs):
#
#   coop-seed            first seed of this shard (required)
#   coop-bucket          gs://bucket where results go (required)
#   coop-run-id          groups the shards of one experiment (required)
#   coop-git-ref         commit/branch/tag to run (default: feature/cooperation-model)
#   coop-repo-url        git URL (default: https://github.com/Dooders/AgentFarm.git)
#   coop-num-replicates  replicates in this shard, seeds seed..seed+n-1 (default: 1)
#   coop-extra-args      extra flags for scripts/run_intrinsic_cooperation_experiment.py
#   coop-omp-threads     OMP_NUM_THREADS for torch (default: leave torch's default)
#   coop-shutdown        "false" keeps the VM up after the run (default: true)
#
# Results land in  <coop-bucket>/intrinsic_cooperation/<coop-run-id>/seed_<coop-seed>/
# with the runner's own output plus host_manifest.json, resources.log,
# startup.log and status.json. The shard is re-run from scratch if the VM
# reboots before status.json says "ok": the runner cannot resume.
#
# Everything runs as root in /opt/agentfarm; these VMs are disposable.
set -euo pipefail

STATE_DIR=${COOP_STATE_DIR:-/var/lib/intrinsic_cooperation}
REPO_DIR=${COOP_REPO_DIR:-/opt/agentfarm}
LOG=${COOP_LOG:-/var/log/intrinsic_cooperation_startup.log}
mkdir -p "$STATE_DIR"
exec > >(tee -a "$LOG") 2>&1

if [[ -f "$STATE_DIR/done" ]]; then
  echo "shard already finished ($(cat "$STATE_DIR/done")); nothing to do"
  exit 0
fi

metadata() {
  local key="$1" default="${2-}"
  local value
  if value=$(curl -sf -H 'Metadata-Flavor: Google' \
      "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$key"); then
    printf '%s' "$value"
  else
    printf '%s' "$default"
  fi
}

SEED=$(metadata coop-seed)
BUCKET=$(metadata coop-bucket)
RUN_ID=$(metadata coop-run-id)
GIT_REF=$(metadata coop-git-ref feature/cooperation-model)
REPO_URL=$(metadata coop-repo-url https://github.com/Dooders/AgentFarm.git)
NUM_REPLICATES=$(metadata coop-num-replicates 1)
EXTRA_ARGS=$(metadata coop-extra-args "")
OMP_THREADS=$(metadata coop-omp-threads "")
SHUTDOWN=$(metadata coop-shutdown true)

for required in SEED BUCKET RUN_ID; do
  if [[ -z "${!required}" ]]; then
    echo "metadata coop-$(echo "$required" | tr '[:upper:]' '[:lower:]') is required" >&2
    exit 2
  fi
done

SHARD="seed_${SEED}"
RESULTS_DIR="$REPO_DIR/results/$SHARD"
REMOTE="${BUCKET%/}/intrinsic_cooperation/$RUN_ID/$SHARD"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
STATUS=failed
EXIT_CODE=1
SYNC_PID=""
SAMPLER_PID=""

sync_results() {
  [[ -d "$RESULTS_DIR" ]] || return 0
  cp -f "$LOG" "$RESULTS_DIR/startup.log" 2>/dev/null || true
  gcloud storage rsync -r --quiet "$RESULTS_DIR" "$REMOTE" || echo "warning: rsync to $REMOTE failed"
}

stop_loop() {
  # The loops are subshells; their current `sleep` child must go too or it
  # keeps the log pipe open after the script exits.
  [[ -n "$1" ]] || return 0
  pkill -P "$1" 2>/dev/null || true
  kill "$1" 2>/dev/null || true
}

finish() {
  stop_loop "$SYNC_PID"
  stop_loop "$SAMPLER_PID"
  if [[ -d "$RESULTS_DIR" ]]; then
    cat > "$RESULTS_DIR/status.json" <<EOF
{"status": "$STATUS", "exit_code": $EXIT_CODE, "seed": $SEED, "num_replicates": $NUM_REPLICATES,
 "run_id": "$RUN_ID", "started_at": "$STARTED_AT", "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
  fi
  sync_results
  if [[ "$STATUS" == "ok" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $REMOTE" > "$STATE_DIR/done"
  fi
  echo "shard $SHARD finished with status=$STATUS exit_code=$EXIT_CODE"
  if [[ "$SHUTDOWN" != "false" ]]; then
    shutdown -h now
  fi
}
trap finish EXIT

echo "=== intrinsic cooperation shard $SHARD (run $RUN_ID, ref $GIT_REF) on $(hostname) at $STARTED_AT"

# --- system packages ------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q git python3 python3-venv python3-dev python3-tk build-essential curl

# --- source at the requested ref ------------------------------------------
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --filter=blob:none "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --quiet origin
git checkout --quiet --detach "$GIT_REF" 2>/dev/null || git checkout --quiet --detach "origin/$GIT_REF"
GIT_COMMIT=$(git rev-parse HEAD)
echo "checked out $GIT_COMMIT"

# --- python environment (CPU torch; the default wheel drags in CUDA) ------
if [[ ! -x venv/bin/python ]]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet -r requirements.txt
pip install --quiet -e .

# --- provenance ------------------------------------------------------------
mkdir -p "$RESULTS_DIR"
MACHINE_TYPE=$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/machine-type || echo unknown)
MACHINE_TYPE=${MACHINE_TYPE##*/}
pip freeze > "$RESULTS_DIR/pip_freeze.txt"
cat > "$RESULTS_DIR/host_manifest.json" <<EOF
{"run_id": "$RUN_ID", "seed": $SEED, "num_replicates": $NUM_REPLICATES,
 "git_ref": "$GIT_REF", "git_commit": "$GIT_COMMIT", "repo_url": "$REPO_URL",
 "hostname": "$(hostname)", "machine_type": "$MACHINE_TYPE", "vcpus": $(nproc),
 "memory_mb": $(free -m | awk '/^Mem:/ {print $2}'), "python": "$(python --version 2>&1)",
 "extra_args": "$EXTRA_ARGS", "omp_threads": "$OMP_THREADS", "started_at": "$STARTED_AT"}
EOF

# --- background: resource samples every minute, partial sync every 10 min ---
(
  echo "time,load1,mem_used_mb,mem_available_mb,swap_used_mb"
  while true; do
    free -m | awk -v t="$(date -u +%H:%M:%S)" -v l="$(cut -d' ' -f1 /proc/loadavg)" \
      '/^Mem:/ {m=$3; a=$7} /^Swap:/ {s=$3} END {printf "%s,%s,%s,%s,%s\n", t, l, m, a, s}'
    sleep 60
  done
) >> "$RESULTS_DIR/resources.log" &
SAMPLER_PID=$!
( while true; do sleep 600; sync_results; done ) &
SYNC_PID=$!

# --- the run ----------------------------------------------------------------
# The experiment script does not set PYTHONHASHSEED itself (run_simulation.py
# does); without it two runs of the same seed are not bit-identical.
export PYTHONHASHSEED=0
if [[ -n "$OMP_THREADS" ]]; then
  export OMP_NUM_THREADS="$OMP_THREADS"
fi
set +e
# shellcheck disable=SC2086
python scripts/run_intrinsic_cooperation_experiment.py \
  --seed "$SEED" \
  --num-replicates "$NUM_REPLICATES" \
  --output-dir "$RESULTS_DIR" \
  $EXTRA_ARGS \
  > "$RESULTS_DIR/stdout.log" 2> "$RESULTS_DIR/stderr.log"
EXIT_CODE=$?
set -e
if [[ $EXIT_CODE -eq 0 && -f "$RESULTS_DIR/intrinsic_cooperation_summary.json" ]]; then
  STATUS=ok
fi
