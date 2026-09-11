#!/usr/bin/env bash
# LOCAL rootless Podman only. No credentials, model execution during builds, or host inference.
set -euo pipefail
if (( $# < 4 )); then
  echo 'Usage: run_isolated_validation.sh IMAGE MODEL.zip DATA.csv|- OUTPUT_DIR [validator options]' >&2
  exit 2
fi
image=$1; model=$(realpath "$2"); data=$3; output=$4; shift 4
[[ $(podman info --format '{{.Host.Security.Rootless}}') == true ]] || { echo 'Rootless Podman required' >&2; exit 2; }
[[ -f "$model" ]] || { echo 'Model ZIP required' >&2; exit 2; }
mkdir -p "$output"; output=$(realpath "$output")
[[ -z $(find "$output" -mindepth 1 -maxdepth 1 -print -quit) ]] || { echo 'Use a new/empty dedicated output directory' >&2; exit 2; }
name="alfie-runtime-validation-$$"
created_id=""
cleanup() {
  if [[ -n "$created_id" ]]; then podman rm -f "$created_id" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
mounts=(--volume "$model:/inputs/model.zip:ro")
options=(--model /inputs/model.zip --output /output)
if [[ "$data" != - ]]; then
  data=$(realpath "$data")
  [[ -f "$data" ]] || { echo 'CSV required' >&2; exit 2; }
  mounts+=(--volume "$data:/inputs/data.csv:ro")
  options+=(--data /inputs/data.csv)
fi
# No host directories other than the individual readonly inputs and dedicated output.
container_id=$(podman create --name "$name" --network=none --http-proxy=false --cap-drop=all \
  --security-opt=no-new-privileges --userns=keep-id:uid=1000,gid=1000 --user=1000:1000 \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g,mode=1777 \
  --cpus=4 --memory=12g --memory-swap=12g --pids-limit=512 \
  --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
  --env NUMBA_NUM_THREADS=4 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache \
  --volume "$output:/output:rw" "${mounts[@]}" \
  "$image" python /app/scripts/validate_runtime.py "${options[@]}" "$@")
# A failed create exits before ownership is recorded: never remove an existing name.
created_id=$container_id
printf '%s\n' "$created_id" > "$output/container-id.txt"
podman inspect "$created_id" > "$output/container-inspect.json"
set +e
timeout --signal=TERM --kill-after=30s 3600 podman start --attach "$created_id"
command_status=$?
set -e
container_status=$(podman inspect --format '{{.State.ExitCode}}' "$created_id")
printf '%s\n' "$command_status" > "$output/attach-exit-code.txt"
printf '%s\n' "$container_status" > "$output/container-exit-code.txt"
if (( command_status != 0 )); then exit "$command_status"; fi
exit "$container_status"
