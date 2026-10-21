#!/usr/bin/env bash
# Run the intrinsic cooperation experiment on Compute Engine, one shard
# (one seed, all four arms) per VM, and collect the results.
#
#   scripts/gcp/intrinsic_cooperation_fleet.sh launch    create one VM per seed
#   scripts/gcp/intrinsic_cooperation_fleet.sh status    VM states + per-shard status.json in the bucket
#   scripts/gcp/intrinsic_cooperation_fleet.sh collect   download the shards and merge them
#   scripts/gcp/intrinsic_cooperation_fleet.sh delete    delete the VMs of this run
#
# Settings come from the environment (defaults in brackets):
#
#   PROJECT         gcloud project                      [gcloud config get-value project]
#   ZONE            Compute Engine zone                 [us-central1-a]
#   BUCKET          gs://bucket for results             (required)
#   RUN_ID          groups the shards of one run        [coop-<UTC timestamp>]
#                   (lowercase letters, digits, hyphens; starts with a letter)
#   SEEDS           seeds to run, one VM each           [101 ... 120]
#   GIT_REF         commit to run on the VMs            [HEAD of this checkout]
#   MACHINE_TYPE    VM type                             [n2-highmem-8]
#   DISK_GB         boot disk size                      [50]
#   PROVISIONING    STANDARD or SPOT                    [STANDARD]
#   NUM_REPLICATES  replicates per VM (seed..seed+n-1)  [1]
#   EXTRA_ARGS      extra runner flags, e.g. "--max-population 200"
#   KEEP_VM         "true" leaves VMs running after the run (for debugging)
#   RESULTS_DIR     where `collect` downloads to        [results/gcp/<RUN_ID>]
#
# The runner cannot resume, so a preempted SPOT VM re-runs its shard from
# scratch on restart (the startup script runs again on every boot).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
STARTUP_SCRIPT="$HERE/intrinsic_cooperation_startup.sh"

PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}
ZONE=${ZONE:-us-central1-a}
BUCKET=${BUCKET:-}
RUN_ID=${RUN_ID:-coop-$(date -u +%Y%m%d-%H%M%S)}
SEEDS=${SEEDS:-$(seq -s ' ' 101 120)}
MACHINE_TYPE=${MACHINE_TYPE:-n2-highmem-8}
DISK_GB=${DISK_GB:-50}
PROVISIONING=${PROVISIONING:-STANDARD}
NUM_REPLICATES=${NUM_REPLICATES:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
KEEP_VM=${KEEP_VM:-false}
RESULTS_DIR=${RESULTS_DIR:-$REPO_ROOT/results/gcp/$RUN_ID}
REMOTE_PREFIX="${BUCKET%/}/intrinsic_cooperation/$RUN_ID"
LABELS="experiment=intrinsic-cooperation,run-id=$(echo "$RUN_ID" | tr '[:upper:]_' '[:lower:]-')"

die() { echo "error: $*" >&2; exit 1; }

require_common() {
  command -v gcloud >/dev/null || die "gcloud CLI not found (https://cloud.google.com/sdk/docs/install)"
  [[ -n "$PROJECT" ]] || die "set PROJECT or run 'gcloud config set project <id>'"
  [[ -n "$BUCKET" ]] || die "set BUCKET=gs://<bucket>"
  [[ "$BUCKET" == gs://* ]] || die "BUCKET must start with gs://"
}

# Instance names must match [a-z]([-a-z0-9]*[a-z0-9])? and be <= 63 chars.
instance_name() { echo "${RUN_ID}-s$1" | tr '[:upper:]_' '[:lower:]-'; }

resolve_git_ref() {
  if [[ -n "${GIT_REF:-}" ]]; then
    echo "$GIT_REF"
    return
  fi
  local head
  head=$(git -C "$REPO_ROOT" rev-parse HEAD)
  if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "warning: working tree has uncommitted changes; VMs run committed $head" >&2
  fi
  if [[ -z "$(git -C "$REPO_ROOT" branch -r --contains "$head" 2>/dev/null)" ]]; then
    die "HEAD $head is not on any remote branch; push it or set GIT_REF"
  fi
  echo "$head"
}

cmd_launch() {
  require_common
  [[ -f "$STARTUP_SCRIPT" ]] || die "missing $STARTUP_SCRIPT"
  local ref
  ref=$(resolve_git_ref)
  gcloud storage ls "$BUCKET" >/dev/null 2>&1 || die "cannot list $BUCKET (does it exist? are you authenticated?)"

  local shutdown=true
  [[ "$KEEP_VM" == "true" ]] && shutdown=false
  local -a provisioning_flags=()
  if [[ "$PROVISIONING" == "SPOT" ]]; then
    provisioning_flags=(--provisioning-model=SPOT --instance-termination-action=STOP)
  fi

  echo "run id:       $RUN_ID"
  echo "project/zone: $PROJECT/$ZONE"
  echo "machine:      $MACHINE_TYPE x $(wc -w <<<"$SEEDS") ($PROVISIONING), ${DISK_GB} GB disk"
  echo "git ref:      $ref"
  echo "results:      $REMOTE_PREFIX/seed_<seed>/"
  [[ -n "$EXTRA_ARGS" ]] && echo "extra args:   $EXTRA_ARGS"

  for seed in $SEEDS; do
    local name
    name=$(instance_name "$seed")
    echo "creating $name (seed $seed)"
    gcloud compute instances create "$name" \
      --project="$PROJECT" --zone="$ZONE" \
      --machine-type="$MACHINE_TYPE" \
      --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
      --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced \
      --scopes=storage-rw,logging-write \
      --labels="$LABELS" \
      "${provisioning_flags[@]}" \
      --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
      --metadata="^;^coop-seed=$seed;coop-bucket=$BUCKET;coop-run-id=$RUN_ID;coop-git-ref=$ref;coop-num-replicates=$NUM_REPLICATES;coop-shutdown=$shutdown;coop-extra-args=$EXTRA_ARGS" \
      --quiet >/dev/null
  done
  echo
  echo "launched. Follow one VM with:"
  echo "  gcloud compute instances tail-serial-port-output $(instance_name "${SEEDS%% *}") --zone=$ZONE"
  echo "and the whole run with:"
  echo "  RUN_ID=$RUN_ID BUCKET=$BUCKET $0 status"
}

cmd_status() {
  require_common
  echo "VMs (label run-id):"
  gcloud compute instances list --project="$PROJECT" \
    --filter="labels.run-id=$(echo "$RUN_ID" | tr '[:upper:]_' '[:lower:]-')" \
    --format="table(name,zone.basename(),machineType.basename(),status,creationTimestamp.date())"
  echo
  echo "shards in $REMOTE_PREFIX:"
  local statuses
  statuses=$(gcloud storage ls "$REMOTE_PREFIX/seed_*/status.json" 2>/dev/null || true)
  if [[ -z "$statuses" ]]; then
    echo "  (none finished yet)"
    return
  fi
  for path in $statuses; do
    printf '  %s  %s\n' "$(basename "$(dirname "$path")")" "$(gcloud storage cat "$path" | tr -d '\n ' )"
  done
}

cmd_collect() {
  require_common
  mkdir -p "$RESULTS_DIR"
  echo "downloading $REMOTE_PREFIX -> $RESULTS_DIR"
  gcloud storage rsync -r "$REMOTE_PREFIX" "$RESULTS_DIR"
  local -a summaries=()
  local failed=0
  for status in "$RESULTS_DIR"/seed_*/status.json; do
    [[ -f "$status" ]] || continue
    local shard
    shard=$(dirname "$status")
    if grep -q '"status": "ok"' "$status"; then
      summaries+=("$shard/intrinsic_cooperation_summary.json")
    else
      echo "shard $(basename "$shard") did not finish: $(cat "$status")"
      failed=1
    fi
  done
  [[ ${#summaries[@]} -gt 0 ]] || die "no finished shards in $RESULTS_DIR"
  local python=${PYTHON:-$REPO_ROOT/venv/bin/python}
  [[ -x "$python" ]] || python=python3
  echo "merging ${#summaries[@]} shards"
  (cd "$REPO_ROOT" && "$python" scripts/merge_intrinsic_cooperation_runs.py "${summaries[@]}" --output-dir "$RESULTS_DIR/merged")
  if [[ $failed -ne 0 ]]; then
    echo "warning: some shards are missing from the merge; see above" >&2
  fi
}

cmd_delete() {
  require_common
  local -a names=()
  for seed in $SEEDS; do names+=("$(instance_name "$seed")"); done
  gcloud compute instances delete "${names[@]}" --project="$PROJECT" --zone="$ZONE" --quiet
}

case "${1:-}" in
  launch) cmd_launch ;;
  status) cmd_status ;;
  collect) cmd_collect ;;
  delete) cmd_delete ;;
  *) awk 'NR > 1 && /^set -euo/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
