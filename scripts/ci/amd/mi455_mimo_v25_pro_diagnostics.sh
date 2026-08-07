#!/usr/bin/env bash
# Capture durable host/container evidence around MiMo-V2.5-Pro startup on MI455.
#
# Run this script as a detached root systemd service so collection survives an
# SSH disconnect. It never removes, stops, restarts, or resets the workload.
# A hard kernel, NIC, storage, or power failure can still lose the final local
# records; use a bastion stream plus platform netconsole/BMC/kdump when possible.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_IMAGE="henryx/xsgl:v0.5.14-gfx1250-rocm-nightlies-20260714-trial-2"
DEFAULT_MODEL_PATH="/model/MiMo-V2.5-Pro"
DEFAULT_PORT="18100"

usage() {
  local program
  program="$(basename -- "$0")"
  cat <<EOF
Usage:
  ${program} run --run-dir ABS --container NAME [options]
  ${program} postmortem --run-dir ABS

Run options:
  --run-id ID                 Safe unique ID (default: UTC timestamp)
  --source PATH               Clean SGLang checkout (default: script checkout)
  --model PATH                Host checkpoint path (default: ${DEFAULT_MODEL_PATH})
  --image IMAGE               Docker image (default: ${DEFAULT_IMAGE})
  --port PORT                 Host-network server port (default: ${DEFAULT_PORT})
  --mode smoke|benchmark      Launcher mode (default: smoke)
  --skip-server-warmup        Isolate loading/init from the first model forward
  --no-rccl-diagnostics       Disable per-process RCCL INFO logs

Set DRY_RUN=1 to print the exact Docker create command without changing state.
For real evidence, pre-create a root-owned parent such as:

  sudo -n install -d -m 0700 /var/lib/sglang-mi455-diagnostics

For a real run, invoke this script through a detached root unit, for example:

  sudo -n systemd-run --no-block --collect --unit=mi455-diag-RUN_ID \
    --property=Type=exec --property=Restart=no --property=KillMode=mixed \
    --property=TimeoutStopSec=15s /absolute/path/${program} run ...

After any reboot, run postmortem before starting another GPU workload. It reads
the exact boot ID saved by run; it never assumes that the failed boot is -1.
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

validate_run_id() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    die "run ID must match [A-Za-z0-9][A-Za-z0-9._-]*"
}

validate_container_name() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
    die "container name contains unsupported characters"
}

validate_absolute_persistent_path() {
  local path="$1"
  [[ -n "${path}" && "${path}" == /* ]] || die "run directory must be absolute"
  [[ "${path}" != *$'\n'* && "${path}" != *$'\r'* ]] || \
    die "run directory must not contain newlines"
  case "/${path#/}/" in
    *"/../"* | *"/./"*) die "run directory must not contain . or .. components" ;;
  esac
  case "${path}" in
    / | /tmp | /tmp/* | /run | /run/* | /var/run | /var/run/* | \
      /dev | /dev/* | /proc | /proc/* | /sys | /sys/*)
      die "run directory must be on persistent storage, not ${path}"
      ;;
  esac
}

validate_port() {
  [[ "$1" =~ ^[0-9]+$ ]] || die "port must be numeric"
  ((10#$1 >= 1 && 10#$1 <= 65535)) || die "port must be between 1 and 65535"
}

validate_interval() {
  local name="$1"
  local value="$2"
  local minimum="$3"
  local maximum="$4"

  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an integer"
  ((10#${value} >= minimum && 10#${value} <= maximum)) || \
    die "${name} must be between ${minimum} and ${maximum} seconds"
}

print_command() {
  printf '%q' "$1"
  shift
  if [[ $# -gt 0 ]]; then
    printf ' %q' "$@"
  fi
  printf '\n'
}

sync_file() {
  python3 -c '
import os
import sys

path = sys.argv[1]
flags = os.O_RDONLY
if os.path.isdir(path) and hasattr(os, "O_DIRECTORY"):
    flags |= os.O_DIRECTORY
descriptor = os.open(path, flags)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
' "$1" 2>/dev/null || sync -d "$1" 2>/dev/null || true
}

read_boot_id() {
  # journalctl accepts the compact machine ID form, while procfs exposes a UUID.
  tr -d '[:space:]-' </proc/sys/kernel/random/boot_id
}

capture_to() {
  local destination="$1"
  shift
  {
    printf 'captured_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'command='
    print_command "$@"
    if "$@"; then
      printf '\nstatus=0\n'
    else
      local status=$?
      printf '\nstatus=%s\n' "${status}"
    fi
  } >"${destination}" 2>&1
  chmod 0600 "${destination}"
  sync_file "${destination}"
}

bounded_stream() {
  local output="$1"
  local max_bytes="$2"
  local file_count="$3"
  local fifo
  local producer_pid
  local producer_status
  local stream_signal=0
  local writer_pid
  local writer_status
  shift 3

  fifo="${output}.pipe.$$"
  mkfifo -m 0600 "${fifo}"
  python3 -c '
import os
import sys
import time
from pathlib import Path

output = Path(sys.argv[1])
max_bytes = int(sys.argv[2])
file_count = int(sys.argv[3])
data_sync = getattr(os, "fdatasync", os.fsync)


def open_output():
    return output.open("xb", buffering=0)


def rotate(file):
    file.flush()
    data_sync(file.fileno())
    file.close()
    oldest = output.with_name(f"{output.name}.{file_count - 1}")
    if oldest.exists():
        oldest.unlink()
    for index in range(file_count - 2, 0, -1):
        source = output.with_name(f"{output.name}.{index}")
        if source.exists():
            os.replace(source, output.with_name(f"{output.name}.{index + 1}"))
    if file_count > 1 and output.exists():
        os.replace(output, output.with_name(f"{output.name}.1"))
    elif output.exists():
        output.unlink()
    return open_output()


file = open_output()
size = 0
bytes_since_sync = 0
last_sync = time.monotonic()
try:
    while True:
        chunk = sys.stdin.buffer.read1(65536)
        if not chunk:
            break
        if size and size + len(chunk) > max_bytes:
            file = rotate(file)
            size = 0
            bytes_since_sync = 0
        file.write(chunk)
        size += len(chunk)
        bytes_since_sync += len(chunk)
        now = time.monotonic()
        if bytes_since_sync >= 1048576 or now - last_sync >= 1:
            data_sync(file.fileno())
            bytes_since_sync = 0
            last_sync = now
finally:
    file.flush()
    data_sync(file.fileno())
    file.close()
' "${output}" "${max_bytes}" "${file_count}" <"${fifo}" &
  writer_pid=$!
  "$@" >"${fifo}" 2>&1 &
  producer_pid=$!

  stop_stream_children() {
    stream_signal=143
    kill -TERM "${producer_pid}" 2>/dev/null || true
    kill -INT "${writer_pid}" 2>/dev/null || true
  }
  trap stop_stream_children TERM INT

  set +e
  while [[ "${stream_signal}" -eq 0 ]] && \
    kill -0 "${producer_pid}" 2>/dev/null && \
    kill -0 "${writer_pid}" 2>/dev/null; do
    if ! jobs -pr | grep -qx "${producer_pid}" || \
      ! jobs -pr | grep -qx "${writer_pid}"; then
      break
    fi
    sleep 1
  done
  if jobs -pr | grep -qx "${producer_pid}" && \
    ! jobs -pr | grep -qx "${writer_pid}"; then
    kill -TERM "${producer_pid}" 2>/dev/null || true
  fi
  local grace=20
  while ((grace > 0)) && { jobs -pr | grep -Eq \
    "^(${producer_pid}|${writer_pid})$"; }; do
    sleep 0.1
    ((grace--)) || true
  done
  local producer_forced=0
  local writer_forced=0
  if jobs -pr | grep -qx "${producer_pid}"; then
    kill -KILL "${producer_pid}" 2>/dev/null || true
    producer_forced=1
  fi
  wait "${producer_pid}"
  producer_status=$?
  if [[ "${producer_forced}" -eq 1 ]]; then
    producer_status=124
  fi
  if jobs -pr | grep -qx "${writer_pid}"; then
    kill -KILL "${writer_pid}" 2>/dev/null || true
    writer_forced=1
  fi
  wait "${writer_pid}"
  writer_status=$?
  if [[ "${writer_forced}" -eq 1 ]]; then
    writer_status=124
  fi
  set -e
  rm -- "${fifo}" 2>/dev/null || true
  trap - TERM INT
  if [[ "${stream_signal}" -ne 0 ]]; then
    return "${stream_signal}"
  elif [[ "${producer_status}" -ne 0 ]]; then
    return "${producer_status}"
  fi
  return "${writer_status}"
}

static_host_snapshot() {
  local path

  printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'hostname=%s\n' "$(hostname 2>/dev/null || printf unknown)"
  printf 'boot_id=%s\n' "$(read_boot_id)"
  printf '\n[uname]\n'
  uname -a || true
  printf '\n[uptime]\n'
  cat /proc/uptime /proc/loadavg 2>/dev/null || true
  printf '\n[kernel command line]\n'
  cat /proc/cmdline 2>/dev/null || true
  printf '\n[amdgpu module parameters]\n'
  for path in /sys/module/amdgpu/parameters/*; do
    [[ -r "${path}" ]] || continue
    printf '%s=' "$(basename -- "${path}")"
    cat "${path}" 2>/dev/null || printf '<unreadable>\n'
  done
  printf '\n[devices]\n'
  ls -l /dev/kfd /dev/dri/renderD* 2>/dev/null || true
  printf '\n[memory]\n'
  free -h 2>/dev/null || true
  cat /proc/meminfo 2>/dev/null || true
  printf '\n[pressure]\n'
  for path in /proc/pressure/*; do
    [[ -r "${path}" ]] || continue
    printf '%s\n' "${path}"
    cat "${path}" || true
  done
  printf '\n[filesystems]\n'
  df -hT 2>/dev/null || true
  printf '\n[network links]\n'
  ip -details -statistics link 2>/dev/null || true
  printf '\n[network routes]\n'
  ip route show table all 2>/dev/null || true
  printf '\n[network neighbours]\n'
  ip neigh show 2>/dev/null || true
  printf '\n[PCI devices]\n'
  timeout --signal=TERM --kill-after=2s 15s lspci -nnk 2>/dev/null || true
  timeout --signal=TERM --kill-after=2s 15s lspci -vv 2>/dev/null || true
  printf '\n[amdgpu module]\n'
  modinfo amdgpu 2>/dev/null || true
  printf '\n[network drivers]\n'
  for path in /sys/class/net/*; do
    [[ -e "${path}" ]] || continue
    printf '%s\n' "${path}"
    timeout --signal=TERM --kill-after=2s 5s \
      ethtool -i "$(basename -- "${path}")" 2>/dev/null || true
    timeout --signal=TERM --kill-after=2s 5s \
      ethtool -S "$(basename -- "${path}")" 2>/dev/null || true
  done
  printf '\n[KFD topology properties]\n'
  for path in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [[ -r "${path}" ]] || continue
    printf '%s\n' "${path}"
    cat "${path}" || true
  done
  printf '\n[pstore]\n'
  find /sys/fs/pstore -maxdepth 1 -type f -printf '%f %s bytes\n' 2>/dev/null || true
  printf '\n[kdump services]\n'
  systemctl --no-pager --full status kdump.service kdump-tools.service 2>/dev/null || true
  printf '\n[journal boots]\n'
  journalctl --list-boots --no-pager 2>/dev/null || true
  printf '\n[last reboots/shutdowns]\n'
  last -x 2>/dev/null | head -80 || true
}

selected_container_inspect() {
  local container="$1"

  docker inspect --format 'id={{.Id}}' "${container}"
  docker inspect --format 'name={{.Name}} created={{.Created}} image={{.Image}}' "${container}"
  docker inspect --format 'path={{json .Path}} args={{json .Args}}' "${container}"
  docker inspect --format 'state={{json .State}}' "${container}"
  docker inspect --format 'restart_count={{.RestartCount}}' "${container}"
  docker inspect --format 'log_config={{json .HostConfig.LogConfig}}' "${container}"
  docker inspect --format 'network_mode={{json .HostConfig.NetworkMode}} ipc_mode={{json .HostConfig.IpcMode}} shm_size={{.HostConfig.ShmSize}}' "${container}"
  docker inspect --format 'devices={{json .HostConfig.Devices}}' "${container}"
  docker inspect --format 'mounts={{json .Mounts}}' "${container}"
}

selected_image_inspect() {
  local image="$1"

  docker image inspect --format 'id={{.Id}}' "${image}"
  docker image inspect --format 'repo_digests={{json .RepoDigests}}' "${image}"
  docker image inspect --format 'created={{.Created}} os={{.Os}} architecture={{.Architecture}}' "${image}"
  docker image inspect --format 'entrypoint={{json .Config.Entrypoint}} cmd={{json .Config.Cmd}}' "${image}"
}

RECORDER_PIDS=()
REQUIRED_RECORDER_NAMES=()
REQUIRED_RECORDER_PIDS=()
RUN_DIRECTORY=""

track_recorder() {
  local name="$1"
  local pid="$2"
  local required="${3:-0}"

  RECORDER_PIDS+=("${pid}")
  printf '%s pid=%s required=%s\n' "${name}" "${pid}" "${required}" \
    >>"${RUN_DIRECTORY}/collector-pids.txt"
  if [[ "${required}" == "1" ]]; then
    REQUIRED_RECORDER_NAMES+=("${name}")
    REQUIRED_RECORDER_PIDS+=("${pid}")
  fi
}

verify_required_recorders() {
  local index
  local failed="0"

  sleep 1
  for ((index = 0; index < ${#REQUIRED_RECORDER_PIDS[@]}; index++)); do
    if ! kill -0 "${REQUIRED_RECORDER_PIDS[index]}" 2>/dev/null; then
      printf 'required_recorder_failed name=%s pid=%s ts=%s\n' \
        "${REQUIRED_RECORDER_NAMES[index]}" \
        "${REQUIRED_RECORDER_PIDS[index]}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >>"${RUN_DIRECTORY}/supervisor.log"
      failed="1"
    fi
  done
  [[ "${failed}" == "0" ]]
}

cleanup_recorders() {
  local live_pids
  local pid
  local remaining=5

  set +e
  live_pids="$(jobs -pr)"
  for pid in "${RECORDER_PIDS[@]:-}"; do
    if printf '%s\n' "${live_pids}" | grep -qx "${pid}"; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  while ((remaining > 0)); do
    [[ -z "$(jobs -pr)" ]] && break
    sleep 1
    ((remaining--)) || true
  done
  live_pids="$(jobs -pr)"
  for pid in "${RECORDER_PIDS[@]:-}"; do
    if printf '%s\n' "${live_pids}" | grep -qx "${pid}"; then
      printf 'collector_force_kill pid=%s ts=%s\n' \
        "${pid}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >>"${RUN_DIRECTORY}/supervisor.log"
      kill -KILL "${pid}" 2>/dev/null || true
    else
      wait "${pid}" 2>/dev/null || true
    fi
  done
  if [[ -n "${RUN_DIRECTORY}" && -d "${RUN_DIRECTORY}" ]]; then
    printf 'recorders_stopped_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >>"${RUN_DIRECTORY}/supervisor.log"
    sync_file "${RUN_DIRECTORY}/supervisor.log"
  fi
}

artifact_flusher() {
  local directory="$1"
  local interval="$2"

  exec python3 - "${directory}" "${interval}" <<'PY'
import os
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
interval = int(sys.argv[2])
data_sync = getattr(os, "fdatasync", os.fsync)
while True:
    for path in root.glob("**/*"):
        if not path.is_file():
            continue
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                data_sync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    for directory in (root / "container", root):
        try:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    time.sleep(interval)
PY
}

host_sampler() {
  local container_id_file="$1"
  local output="$2"
  local interval="$3"
  local evidence_directory
  local path

  evidence_directory="$(dirname -- "${output}")"
  while true; do
    printf 'ts=%s uptime=' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    tr '\n' ' ' </proc/uptime 2>/dev/null || true
    printf '\n'
    printf 'loadavg='
    cat /proc/loadavg 2>/dev/null || true
    grep -E '^(MemAvailable|MemFree|Buffers|Cached|SwapFree|Dirty|Writeback):' \
      /proc/meminfo 2>/dev/null || true
    for path in /proc/pressure/cpu /proc/pressure/io /proc/pressure/memory; do
      [[ -r "${path}" ]] || continue
      printf '%s=' "$(basename -- "${path}")"
      tr '\n' ';' <"${path}"
      printf '\n'
    done
    printf '[network]\n'
    timeout --signal=TERM --kill-after=1s 3s \
      ip -statistics link 2>/dev/null || true
    timeout --signal=TERM --kill-after=1s 3s \
      ip neigh show 2>/dev/null || true
    printf '[evidence storage]\n'
    timeout --signal=TERM --kill-after=1s 3s \
      df -Pk "${evidence_directory}" 2>/dev/null || true
    timeout --signal=TERM --kill-after=1s 3s \
      df -Pi "${evidence_directory}" 2>/dev/null || true
    timeout --signal=TERM --kill-after=1s 3s \
      du -sk "${evidence_directory}" 2>/dev/null || true
    if [[ -s "${container_id_file}" ]]; then
      timeout --signal=TERM --kill-after=1s 3s \
        docker inspect --format 'container_state={{json .State}}' \
        "$(<"${container_id_file}")" 2>&1 || true
    else
      printf 'container_state=<not-created>\n'
    fi
    printf '%s\n' '---'
    sleep "${interval}"
  done >>"${output}" 2>&1
}

collector_health_sampler() {
  local pid_file="$1"
  local output="$2"
  local interval="$3"
  local name
  local pid
  local required

  while true; do
    printf 'ts=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    while read -r name pid required; do
      pid="${pid#pid=}"
      required="${required#required=}"
      [[ "${pid}" =~ ^[0-9]+$ ]] || continue
      if kill -0 "${pid}" 2>/dev/null; then
        printf 'name=%s pid=%s required=%s alive=yes\n' "${name}" "${pid}" "${required}"
      else
        printf 'name=%s pid=%s required=%s alive=no\n' "${name}" "${pid}" "${required}"
      fi
    done <"${pid_file}"
    printf '%s\n' '---'
    sleep "${interval}"
  done >>"${output}" 2>&1
}

health_sampler() {
  local port="$1"
  local output="$2"
  local interval="$3"

  while true; do
    printf 'ts=%s ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if command -v curl >/dev/null 2>&1; then
      curl --silent --show-error --max-time 2 --output /dev/null \
        --write-out 'http_code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
        "http://127.0.0.1:${port}/health" 2>&1 || printf 'health_request_failed\n'
    else
      printf 'curl_unavailable\n'
    fi
    sleep "${interval}"
  done >>"${output}" 2>&1
}

gpu_sysfs_sampler() {
  local output="$1"
  local interval="$2"
  local path

  while true; do
    printf 'ts=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for path in \
      /sys/class/drm/card*/device/gpu_busy_percent \
      /sys/class/drm/card*/device/mem_info_vram_used \
      /sys/class/drm/card*/device/mem_info_vram_total \
      /sys/class/drm/card*/device/hwmon/hwmon*/temp*_input \
      /sys/class/drm/card*/device/hwmon/hwmon*/power*_average \
      /sys/class/drm/card*/device/ras/*_err_count \
      /sys/bus/pci/devices/*/a[e]r_dev_correctable \
      /sys/bus/pci/devices/*/a[e]r_dev_nonfatal \
      /sys/bus/pci/devices/*/a[e]r_dev_fatal; do
      [[ -r "${path}" ]] || continue
      printf '%s=' "${path}"
      cat "${path}" 2>&1 || true
    done
    printf '%s\n' '---'
    sleep "${interval}"
  done >>"${output}" 2>&1
}

journal_sync_sampler() {
  local output="$1"
  local interval="$2"

  while true; do
    printf 'ts=%s ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if timeout --signal=TERM --kill-after=2s 10s journalctl --sync; then
      printf 'status=0\n'
    else
      printf 'status=%s\n' "$?"
    fi
    sleep "${interval}"
  done >>"${output}" 2>&1
}

parse_run_options() {
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIRECTORY=""
  CONTAINER_NAME=""
  SOURCE_PATH="${REPO_ROOT}"
  MODEL_PATH="${DEFAULT_MODEL_PATH}"
  IMAGE="${DEFAULT_IMAGE}"
  PORT="${DEFAULT_PORT}"
  MODE="smoke"
  SKIP_WARMUP="0"
  ENABLE_RCCL="1"
  HOST_INTERVAL="${DIAG_HOST_INTERVAL:-5}"
  HEALTH_INTERVAL="${DIAG_HEALTH_INTERVAL:-5}"
  FLUSH_INTERVAL="${DIAG_FLUSH_INTERVAL:-5}"
  GPU_INTERVAL="${DIAG_GPU_INTERVAL:-30}"
  JOURNAL_SYNC_INTERVAL="${DIAG_JOURNAL_SYNC_INTERVAL:-15}"
  MIN_FREE_BYTES="${DIAG_MIN_FREE_BYTES:-5368709120}"
  STREAM_LOG_MAX_BYTES="${DIAG_LOG_MAX_BYTES:-104857600}"
  STREAM_LOG_FILE_COUNT="${DIAG_LOG_FILE_COUNT:-5}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-dir) [[ $# -ge 2 ]] || die "--run-dir requires a value"; RUN_DIRECTORY="$2"; shift 2 ;;
      --run-id) [[ $# -ge 2 ]] || die "--run-id requires a value"; RUN_ID="$2"; shift 2 ;;
      --container) [[ $# -ge 2 ]] || die "--container requires a value"; CONTAINER_NAME="$2"; shift 2 ;;
      --source) [[ $# -ge 2 ]] || die "--source requires a value"; SOURCE_PATH="$2"; shift 2 ;;
      --model) [[ $# -ge 2 ]] || die "--model requires a value"; MODEL_PATH="$2"; shift 2 ;;
      --image) [[ $# -ge 2 ]] || die "--image requires a value"; IMAGE="$2"; shift 2 ;;
      --port) [[ $# -ge 2 ]] || die "--port requires a value"; PORT="$2"; shift 2 ;;
      --mode) [[ $# -ge 2 ]] || die "--mode requires a value"; MODE="$2"; shift 2 ;;
      --skip-server-warmup) SKIP_WARMUP="1"; shift ;;
      --no-rccl-diagnostics) ENABLE_RCCL="0"; shift ;;
      -h | --help) usage; exit 0 ;;
      *) die "unknown run option: $1" ;;
    esac
  done

  [[ -n "${RUN_DIRECTORY}" ]] || die "--run-dir is required"
  [[ -n "${CONTAINER_NAME}" ]] || die "--container is required"
  RUN_DIRECTORY="${RUN_DIRECTORY%/}"
  validate_absolute_persistent_path "${RUN_DIRECTORY}"
  validate_run_id "${RUN_ID}"
  validate_container_name "${CONTAINER_NAME}"
  validate_port "${PORT}"
  [[ "${SOURCE_PATH}" == /* ]] || die "source path must be absolute"
  [[ "${MODEL_PATH}" == /* ]] || die "model path must be absolute"
  [[ "${SOURCE_PATH}" != *$'\n'* && "${MODEL_PATH}" != *$'\n'* ]] || \
    die "source and model paths must not contain newlines"
  [[ "${MODE}" == "smoke" || "${MODE}" == "benchmark" ]] || \
    die "mode must be smoke or benchmark"
  [[ -n "${IMAGE}" && "${IMAGE}" != -* ]] || die "invalid image"
  [[ "${DRY_RUN:-0}" == "0" || "${DRY_RUN:-0}" == "1" ]] || \
    die "DRY_RUN must be 0 or 1"
  validate_interval DIAG_HOST_INTERVAL "${HOST_INTERVAL}" 2 300
  validate_interval DIAG_HEALTH_INTERVAL "${HEALTH_INTERVAL}" 2 300
  validate_interval DIAG_FLUSH_INTERVAL "${FLUSH_INTERVAL}" 5 300
  validate_interval DIAG_GPU_INTERVAL "${GPU_INTERVAL}" 10 600
  validate_interval DIAG_JOURNAL_SYNC_INTERVAL "${JOURNAL_SYNC_INTERVAL}" 10 600
  [[ "${MIN_FREE_BYTES}" =~ ^[0-9]+$ ]] || die "DIAG_MIN_FREE_BYTES must be an integer"
  [[ "${STREAM_LOG_MAX_BYTES}" =~ ^[0-9]+$ ]] || die "DIAG_LOG_MAX_BYTES must be an integer"
  ((STREAM_LOG_MAX_BYTES >= 1048576 && STREAM_LOG_MAX_BYTES <= 1073741824)) || \
    die "DIAG_LOG_MAX_BYTES must be between 1 MiB and 1 GiB"
  [[ "${STREAM_LOG_FILE_COUNT}" =~ ^[0-9]+$ ]] || die "DIAG_LOG_FILE_COUNT must be an integer"
  ((STREAM_LOG_FILE_COUNT >= 1 && STREAM_LOG_FILE_COUNT <= 20)) || \
    die "DIAG_LOG_FILE_COUNT must be between 1 and 20"
}

build_docker_command() {
  local source_head="$1"
  local container_diagnostics="${RUN_DIRECTORY}/container"

  DOCKER_COMMAND=(
    docker create
    --name "${CONTAINER_NAME}"
    --restart=no
    --network=host
    --ipc=host
    --shm-size=32g
    --ulimit memlock=-1:-1
    --ulimit stack=67108864
    --ulimit core=0:0
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    --log-driver local
    --log-opt max-size=100m
    --log-opt max-file=5
    --label "com.sglang.mi455.run_id=${RUN_ID}"
    --label "com.sglang.mi455.source_head=${source_head}"
    -e "MODE=${MODE}"
    -e "MODEL_PATH=/model/MiMo-V2.5-Pro"
    -e "TP_SIZE=4"
    -e "PORT=${PORT}"
    -e "CUDA_VISIBLE_DEVICES=0,1,2,3"
    -e "SGLANG_USE_AITER=0"
    -e "ENABLE_CK=0"
    -e "PYTHONDONTWRITEBYTECODE=1"
    -e "DIAGNOSTICS_DIR=/diagnostics"
    -e "DIAGNOSTICS_RUN_ID=${RUN_ID}"
    -e "ENABLE_RCCL_DIAGNOSTICS=${ENABLE_RCCL}"
    -e "RCCL_DIAGNOSTICS_TO_STDOUT=1"
    -e "CAPTURE_DIAGNOSTICS_SERVER_LOG=0"
    -e "SKIP_SERVER_WARMUP=${SKIP_WARMUP}"
    -v "${SOURCE_PATH}:/workspace/sglang-pr11:ro"
    -v "${MODEL_PATH}:/model/MiMo-V2.5-Pro:ro"
    -v "${container_diagnostics}:/diagnostics:rw"
    -w /workspace/sglang-pr11
    --entrypoint /bin/bash
    "${IMAGE}"
    -lc 'exec scripts/ci/amd/mi455_mimo_v25_pro.sh'
  )
}

write_run_manifest() {
  local source_head="$1"
  local manifest_tmp="${RUN_DIRECTORY}/manifest.tmp"

  {
    printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'run_id=%q\n' "${RUN_ID}"
    printf 'boot_id=%s\n' "$(<"${RUN_DIRECTORY}/boot_id")"
    printf 'container=%q\n' "${CONTAINER_NAME}"
    printf 'source=%q\n' "${SOURCE_PATH}"
    printf 'source_head=%s\n' "${source_head}"
    printf 'model=%q\n' "${MODEL_PATH}"
    printf 'image=%q\n' "${IMAGE}"
    printf 'port=%s\n' "${PORT}"
    printf 'mode=%s\n' "${MODE}"
    printf 'skip_server_warmup=%s\n' "${SKIP_WARMUP}"
    printf 'enable_rccl_diagnostics=%s\n' "${ENABLE_RCCL}"
    printf 'docker_create='
    print_command "${DOCKER_COMMAND[@]}"
  } >"${manifest_tmp}"
  chmod 0600 "${manifest_tmp}"
  sync_file "${manifest_tmp}"
  mv -- "${manifest_tmp}" "${RUN_DIRECTORY}/manifest.txt"
  sync_file "${RUN_DIRECTORY}"
}

run_command() {
  local container_exit
  local container_id
  local create_output
  local create_status
  local source_head="unknown"
  local start_timestamp
  local start_status

  parse_run_options "$@"
  if [[ -d "${SOURCE_PATH}/.git" || -f "${SOURCE_PATH}/.git" ]]; then
    source_head="$(git -c safe.directory="${SOURCE_PATH}" -C "${SOURCE_PATH}" \
      rev-parse HEAD 2>/dev/null || printf unknown)"
  fi
  build_docker_command "${source_head}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'run_dir=%q\n' "${RUN_DIRECTORY}"
    printf 'docker_create='
    print_command "${DOCKER_COMMAND[@]}"
    exit 0
  fi

  [[ "${EUID}" -eq 0 ]] || die "run must execute as root (use sudo -n systemd-run)"
  require_command docker
  require_command flock
  require_command journalctl
  require_command ip
  require_command git
  require_command python3
  require_command mkfifo
  require_command timeout
  require_command fuser
  require_command realpath
  require_command findmnt
  require_command stat
  [[ -d "${SOURCE_PATH}" ]] || die "source checkout not found: ${SOURCE_PATH}"
  [[ -r "${MODEL_PATH}/config.json" ]] || die "model config not found"
  [[ -r "${MODEL_PATH}/model.safetensors.index.json" ]] || die "model index not found"
  git -c safe.directory="${SOURCE_PATH}" -C "${SOURCE_PATH}" \
    rev-parse --verify HEAD >/dev/null 2>&1 || \
    die "source path is not a Git checkout"
  if [[ -n "$(git -c safe.directory="${SOURCE_PATH}" -C "${SOURCE_PATH}" \
    status --porcelain=v1)" ]]; then
    die "source checkout must be clean"
  fi
  source_head="$(git -c safe.directory="${SOURCE_PATH}" -C "${SOURCE_PATH}" rev-parse HEAD)"
  build_docker_command "${source_head}"
  docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "Docker image is unavailable: ${IMAGE}"
  if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    die "container already exists; refusing to replace it: ${CONTAINER_NAME}"
  fi
  if command -v ss >/dev/null 2>&1 && \
    ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
    die "TCP port ${PORT} is already listening"
  fi
  [[ -e /dev/kfd ]] || die "/dev/kfd is missing"
  if command -v fuser >/dev/null 2>&1; then
    local device_nodes=(/dev/kfd)
    local device_path
    local holder_output
    local holder_status
    for device_path in /dev/dri/renderD*; do
      [[ -e "${device_path}" ]] && device_nodes+=("${device_path}")
    done
    set +e
    holder_output="$(timeout --signal=TERM --kill-after=2s 10s \
      fuser "${device_nodes[@]}" 2>&1)"
    holder_status=$?
    set -e
    case "${holder_status}" in
      0) die "GPU device nodes are already in use: ${holder_output}" ;;
      1) ;;
      *) die "could not establish GPU-device isolation (fuser status ${holder_status})" ;;
    esac
  fi
  if [[ -e "${RUN_DIRECTORY}" || -L "${RUN_DIRECTORY}" ]]; then
    die "run directory already exists; refusing to overwrite it"
  fi

  local run_parent
  local canonical_run_directory
  local storage_fstype
  local parent_mode
  local parent_owner
  local available_bytes
  local available_inodes
  run_parent="$(dirname -- "${RUN_DIRECTORY}")"
  [[ -d "${run_parent}" && ! -L "${run_parent}" ]] || \
    die "run-directory parent must be a pre-existing real directory"
  [[ "$(realpath -e -- "${run_parent}")" == "${run_parent}" ]] || \
    die "run-directory parent must not traverse symlinks or aliases"
  parent_owner="$(stat -c %u -- "${run_parent}")"
  parent_mode="$(stat -c %a -- "${run_parent}")"
  [[ "${parent_owner}" == "0" ]] || die "run-directory parent must be owned by root"
  (( (8#${parent_mode} & 8#022) == 0 )) || \
    die "run-directory parent must not be writable by group or other"
  canonical_run_directory="$(realpath -m -- "${RUN_DIRECTORY}")"
  [[ "${canonical_run_directory}" == "${RUN_DIRECTORY}" ]] || \
    die "run directory must not traverse symlinks or aliases"
  storage_fstype="$(findmnt -T "${run_parent}" -n -o FSTYPE 2>/dev/null || true)"
  case "${storage_fstype}" in
    "" | tmpfs | devtmpfs | proc | sysfs | overlay | nfs* | cifs | fuse* | autofs)
      die "run directory filesystem is not suitable for persistent evidence: ${storage_fstype:-unknown}"
      ;;
  esac
  available_bytes="$(df -B1 --output=avail "${run_parent}" | tail -1 | tr -d '[:space:]')"
  available_inodes="$(df --output=iavail "${run_parent}" | tail -1 | tr -d '[:space:]')"
  [[ "${available_bytes}" =~ ^[0-9]+$ && "${available_inodes}" =~ ^[0-9]+$ ]] || \
    die "could not determine evidence-filesystem capacity"
  ((available_bytes >= MIN_FREE_BYTES)) || \
    die "evidence filesystem has less than ${MIN_FREE_BYTES} bytes free"
  ((available_inodes >= 10000)) || die "evidence filesystem has fewer than 10000 free inodes"
  mkdir -- "${RUN_DIRECTORY}"
  chmod 0700 "${RUN_DIRECTORY}"
  mkdir -- "${RUN_DIRECTORY}/container"
  chmod 0700 "${RUN_DIRECTORY}/container"
  umask 077
  exec 9>"${RUN_DIRECTORY}/.lock"
  flock -n 9 || die "another recorder holds the run-directory lock"
  printf '%s\n' "$(read_boot_id)" \
    >"${RUN_DIRECTORY}/boot_id"
  printf '%s\n' "${CONTAINER_NAME}" >"${RUN_DIRECTORY}/container.name"
  printf 'active_since=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"${RUN_DIRECTORY}/active"
  : >"${RUN_DIRECTORY}/collector-pids.txt"
  : >"${RUN_DIRECTORY}/supervisor.log"
  chmod 0600 \
    "${RUN_DIRECTORY}/boot_id" \
    "${RUN_DIRECTORY}/container.name" \
    "${RUN_DIRECTORY}/active" \
    "${RUN_DIRECTORY}/collector-pids.txt" \
    "${RUN_DIRECTORY}/supervisor.log" \
    "${RUN_DIRECTORY}/.lock"
  write_run_manifest "${source_head}"
  printf 'fstype=%s available_bytes=%s available_inodes=%s\n' \
    "${storage_fstype}" "${available_bytes}" "${available_inodes}" \
    >"${RUN_DIRECTORY}/storage.txt"

  trap cleanup_recorders EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  bounded_stream "${RUN_DIRECTORY}/kernel-follow.log" \
    "${STREAM_LOG_MAX_BYTES}" "${STREAM_LOG_FILE_COUNT}" \
    journalctl -k -b "$(<"${RUN_DIRECTORY}/boot_id")" -n 0 -f \
    --no-pager -o short-precise 9>&- &
  track_recorder kernel-follow "$!" 1

  bounded_stream "${RUN_DIRECTORY}/docker-services-follow.log" \
    "${STREAM_LOG_MAX_BYTES}" "${STREAM_LOG_FILE_COUNT}" \
    journalctl -b "$(<"${RUN_DIRECTORY}/boot_id")" -n 0 -f \
    -u docker.service -u containerd.service --no-pager -o short-precise 9>&- &
  track_recorder docker-services-follow "$!"

  bounded_stream "${RUN_DIRECTORY}/docker-events.log" \
    "${STREAM_LOG_MAX_BYTES}" "${STREAM_LOG_FILE_COUNT}" \
    docker events --since "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --filter "container=${CONTAINER_NAME}" --format '{{json .}}' 9>&- &
  track_recorder docker-events "$!" 1

  if command -v vmstat >/dev/null 2>&1; then
    bounded_stream "${RUN_DIRECTORY}/vmstat.log" \
      "${STREAM_LOG_MAX_BYTES}" "${STREAM_LOG_FILE_COUNT}" vmstat -t 5 9>&- &
    track_recorder vmstat "$!"
  fi
  bounded_stream "${RUN_DIRECTORY}/ip-monitor.log" \
    "${STREAM_LOG_MAX_BYTES}" "${STREAM_LOG_FILE_COUNT}" \
    ip -ts monitor link neigh 9>&- &
  track_recorder ip-monitor "$!"
  host_sampler "${RUN_DIRECTORY}/container.id" "${RUN_DIRECTORY}/host-samples.log" \
    "${HOST_INTERVAL}" 9>&- &
  track_recorder host-sampler "$!" 1
  health_sampler "${PORT}" "${RUN_DIRECTORY}/health.log" \
    "${HEALTH_INTERVAL}" 9>&- &
  track_recorder health-sampler "$!"
  gpu_sysfs_sampler "${RUN_DIRECTORY}/gpu-sysfs.log" "${GPU_INTERVAL}" 9>&- &
  track_recorder gpu-sysfs "$!"
  journal_sync_sampler "${RUN_DIRECTORY}/journal-sync.log" \
    "${JOURNAL_SYNC_INTERVAL}" 9>&- &
  track_recorder journal-sync "$!"
  artifact_flusher "${RUN_DIRECTORY}" "${FLUSH_INTERVAL}" 9>&- &
  track_recorder artifact-flusher "$!" 1
  collector_health_sampler "${RUN_DIRECTORY}/collector-pids.txt" \
    "${RUN_DIRECTORY}/collector-health.log" "${HOST_INTERVAL}" 9>&- &
  track_recorder collector-health "$!"

  if ! verify_required_recorders; then
    printf 'failed_at=%s stage=recorder_preflight status=2\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${RUN_DIRECTORY}/failed"
    exit 2
  fi
  if command -v logger >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=1s 5s logger -t sglang-mi455-diagnostics \
      "run_id=${RUN_ID} phase=collectors_ready boot_id=$(<"${RUN_DIRECTORY}/boot_id")" \
      || true
  fi
  if [[ -w /dev/kmsg ]]; then
    printf '<6>sglang-mi455-diagnostics run_id=%s phase=collectors_ready\n' \
      "${RUN_ID}" >/dev/kmsg || true
  fi
  printf 'collectors_ready_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"${RUN_DIRECTORY}/supervisor.log"
  sync_file "${RUN_DIRECTORY}/supervisor.log"
  if ! timeout --signal=TERM --kill-after=2s 10s journalctl --sync; then
    printf 'collectors_ready_journal_sync_failed ts=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${RUN_DIRECTORY}/supervisor.log"
    sync_file "${RUN_DIRECTORY}/supervisor.log"
  fi

  capture_to "${RUN_DIRECTORY}/host-before.txt" static_host_snapshot
  capture_to "${RUN_DIRECTORY}/kernel-before.log" \
    journalctl -k -b "$(<"${RUN_DIRECTORY}/boot_id")" --no-pager -o short-precise
  capture_to "${RUN_DIRECTORY}/docker-services-before.log" \
    journalctl -b "$(<"${RUN_DIRECTORY}/boot_id")" -u docker.service \
    -u containerd.service --no-pager -o short-precise
  capture_to "${RUN_DIRECTORY}/image.txt" selected_image_inspect "${IMAGE}"
  capture_to "${RUN_DIRECTORY}/source.txt" git \
    -c safe.directory="${SOURCE_PATH}" -C "${SOURCE_PATH}" status --short --branch
  capture_to "${RUN_DIRECTORY}/model-hashes.txt" sha256sum \
    "${MODEL_PATH}/config.json" "${MODEL_PATH}/model.safetensors.index.json"
  if command -v rocm-smi >/dev/null 2>&1; then
    capture_to "${RUN_DIRECTORY}/rocm-smi-before.txt" timeout \
      --signal=TERM --kill-after=2s 20s rocm-smi
  elif command -v amd-smi >/dev/null 2>&1; then
    capture_to "${RUN_DIRECTORY}/amd-smi-before.txt" timeout \
      --signal=TERM --kill-after=2s 20s amd-smi static
  else
    printf 'status=unavailable command=rocm-smi_or_amd-smi\n' \
      >"${RUN_DIRECTORY}/gpu-smi-before.txt"
  fi

  printf 'docker_create_started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"${RUN_DIRECTORY}/supervisor.log"
  set +e
  create_output="$("${DOCKER_COMMAND[@]}" 2>>"${RUN_DIRECTORY}/docker-create.stderr")"
  create_status=$?
  set -e
  printf 'docker_create_status=%s\n' "${create_status}" >>"${RUN_DIRECTORY}/supervisor.log"
  if [[ "${create_status}" -ne 0 ]]; then
    printf 'failed_at=%s stage=docker_create status=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${create_status}" \
      >"${RUN_DIRECTORY}/failed"
    exit "${create_status}"
  fi
  container_id="${create_output##*$'\n'}"
  [[ "${container_id}" =~ ^[a-f0-9]{12,64}$ ]] || die "Docker returned an invalid container ID"
  printf '%s\n' "${container_id}" >"${RUN_DIRECTORY}/container.id"
  sync_file "${RUN_DIRECTORY}/container.id"
  capture_to "${RUN_DIRECTORY}/container-created.txt" \
    selected_container_inspect "${container_id}"

  start_timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'docker_start_at=%s\n' "${start_timestamp}" \
    >>"${RUN_DIRECTORY}/supervisor.log"
  if command -v logger >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=1s 5s logger -t sglang-mi455-diagnostics \
      "run_id=${RUN_ID} phase=docker_start container_id=${container_id}" || true
  fi
  if [[ -w /dev/kmsg ]]; then
    printf '<6>sglang-mi455-diagnostics run_id=%s phase=docker_start container_id=%s\n' \
      "${RUN_ID}" "${container_id}" >/dev/kmsg || true
  fi
  sync_file "${RUN_DIRECTORY}/supervisor.log"
  if ! timeout --signal=TERM --kill-after=2s 10s journalctl --sync; then
    printf 'marker_journal_sync_failed ts=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${RUN_DIRECTORY}/supervisor.log"
    sync_file "${RUN_DIRECTORY}/supervisor.log"
  fi
  set +e
  docker start "${container_id}" >>"${RUN_DIRECTORY}/docker-start.stdout" \
    2>>"${RUN_DIRECTORY}/docker-start.stderr"
  start_status=$?
  set -e
  printf 'docker_start_status=%s\n' "${start_status}" >>"${RUN_DIRECTORY}/supervisor.log"
  if [[ "${start_status}" -ne 0 ]]; then
    capture_to "${RUN_DIRECTORY}/container-start-failed.txt" \
      selected_container_inspect "${container_id}"
    printf 'failed_at=%s stage=docker_start status=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${start_status}" \
      >"${RUN_DIRECTORY}/failed"
    exit "${start_status}"
  fi

  bounded_stream "${RUN_DIRECTORY}/container-stdout-stderr.log" \
    "${STREAM_LOG_MAX_BYTES}" "${STREAM_LOG_FILE_COUNT}" \
    docker logs --follow --timestamps --since "${start_timestamp}" \
    "${container_id}" 9>&- &
  local docker_logs_pid=$!
  track_recorder docker-logs "${docker_logs_pid}"
  sleep 1
  if [[ "$(docker inspect --format '{{.State.Running}}' "${container_id}" 2>/dev/null || true)" == "true" ]] && \
    ! kill -0 "${docker_logs_pid}" 2>/dev/null; then
    printf 'optional_recorder_failed name=docker-logs pid=%s ts=%s\n' \
      "${docker_logs_pid}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >>"${RUN_DIRECTORY}/supervisor.log"
  fi

  printf 'docker_wait_started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"${RUN_DIRECTORY}/supervisor.log"
  container_exit="$(docker wait "${container_id}")"
  [[ "${container_exit}" =~ ^[0-9]+$ ]] || container_exit=125
  capture_to "${RUN_DIRECTORY}/container-final.txt" \
    selected_container_inspect "${container_id}"
  capture_to "${RUN_DIRECTORY}/container-final.log" \
    docker logs --timestamps "${container_id}"
  printf 'completed_at=%s container_exit=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${container_exit}" \
    >"${RUN_DIRECTORY}/complete"
  mv -- "${RUN_DIRECTORY}/active" "${RUN_DIRECTORY}/active.completed"
  sync_file "${RUN_DIRECTORY}/complete"
  sync_file "${RUN_DIRECTORY}"
  exit "${container_exit}"
}

postmortem_command() {
  local current_boot_id
  local journal_boot_available="no"
  local output_directory
  local recovery_directory
  local run_directory=""
  local saved_boot_id
  local container_id=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-dir) [[ $# -ge 2 ]] || die "--run-dir requires a value"; run_directory="$2"; shift 2 ;;
      -h | --help) usage; exit 0 ;;
      *) die "unknown postmortem option: $1" ;;
    esac
  done
  [[ -n "${run_directory}" ]] || die "--run-dir is required"
  run_directory="${run_directory%/}"
  validate_absolute_persistent_path "${run_directory}"
  [[ "${EUID}" -eq 0 ]] || die "postmortem must execute as root"
  require_command journalctl
  require_command sha256sum
  require_command flock
  require_command python3
  require_command realpath
  require_command stat
  [[ -d "${run_directory}" && ! -L "${run_directory}" ]] || \
    die "run directory not found or is a symlink: ${run_directory}"
  [[ "$(realpath -e -- "${run_directory}")" == "${run_directory}" ]] || \
    die "run directory must not traverse symlinks or aliases"
  [[ "$(stat -c %u -- "${run_directory}")" == "0" ]] || \
    die "run directory must be owned by root"
  [[ -r "${run_directory}/boot_id" ]] || die "saved boot ID is missing"
  [[ -r "${run_directory}/.lock" && ! -L "${run_directory}/.lock" ]] || \
    die "recorder lock is missing or unsafe"
  exec 8<"${run_directory}/.lock"
  flock -n 8 || die "the recorder is still active; refusing an inconsistent postmortem"

  saved_boot_id="$(tr -d '[:space:]-' <"${run_directory}/boot_id")"
  current_boot_id="$(read_boot_id)"
  [[ "${saved_boot_id}" =~ ^[a-f0-9]{32}$ ]] || die "saved boot ID is invalid"
  if journalctl --list-boots --no-pager 2>/dev/null | grep -Fq "${saved_boot_id}"; then
    journal_boot_available="yes"
  fi
  recovery_directory="${run_directory}/recovery"
  if [[ -L "${recovery_directory}" ]]; then
    die "recovery path must not be a symlink"
  elif [[ -e "${recovery_directory}" && ! -d "${recovery_directory}" ]]; then
    die "recovery path exists but is not a directory"
  elif [[ ! -e "${recovery_directory}" ]]; then
    mkdir -- "${recovery_directory}"
  fi
  chmod 0700 "${recovery_directory}"
  output_directory="${recovery_directory}/$(date -u +%Y%m%dT%H%M%SZ)"
  [[ ! -e "${output_directory}" && ! -L "${output_directory}" ]] || \
    die "postmortem output already exists"
  mkdir -- "${output_directory}"
  chmod 0700 "${output_directory}"
  umask 077

  {
    printf 'captured_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'saved_boot_id=%s\n' "${saved_boot_id}"
    printf 'current_boot_id=%s\n' "${current_boot_id}"
    printf 'saved_boot_in_journal=%s\n' "${journal_boot_available}"
    if [[ "${saved_boot_id}" == "${current_boot_id}" ]]; then
      printf 'reboot_observed=no\n'
      printf 'note=the recorder boot is still current; this is not previous-boot evidence\n'
    else
      printf 'reboot_observed=yes\n'
    fi
  } >"${output_directory}/summary.txt"

  capture_to "${output_directory}/journal-boots.txt" journalctl --list-boots --no-pager
  capture_to "${output_directory}/saved-boot-kernel.log" \
    journalctl -k -b "${saved_boot_id}" --no-pager -o short-precise
  capture_to "${output_directory}/saved-boot-full.log" \
    journalctl -b "${saved_boot_id}" --no-pager -o short-precise
  capture_to "${output_directory}/saved-boot-docker.log" \
    journalctl -b "${saved_boot_id}" -u docker.service -u containerd.service \
    --no-pager -o short-precise
  capture_to "${output_directory}/current-host.txt" static_host_snapshot
  capture_to "${output_directory}/last-x.txt" last -x
  if command -v coredumpctl >/dev/null 2>&1; then
    capture_to "${output_directory}/coredumps.txt" coredumpctl --no-pager list \
      "_BOOT_ID=${saved_boot_id}"
  else
    printf 'status=unavailable command=coredumpctl\n' >"${output_directory}/coredumps.txt"
  fi
  capture_to "${output_directory}/kdump.txt" systemctl --no-pager --full \
    status kdump.service kdump-tools.service
  if command -v ras-mc-ctl >/dev/null 2>&1; then
    capture_to "${output_directory}/rasdaemon.txt" ras-mc-ctl --errors
  else
    printf 'status=unavailable command=ras-mc-ctl\n' \
      >"${output_directory}/rasdaemon.txt"
  fi

  mkdir -- "${output_directory}/pstore"
  if [[ -d /sys/fs/pstore ]] && cp -a /sys/fs/pstore/. "${output_directory}/pstore/" \
    2>"${output_directory}/pstore-copy.stderr"; then
    printf 'status=0\n' >"${output_directory}/pstore-copy.status"
  else
    printf 'status=unavailable_or_failed\n' >"${output_directory}/pstore-copy.status"
  fi
  chmod -R go-rwx "${output_directory}/pstore"

  if [[ -s "${run_directory}/container.id" ]]; then
    container_id="$(tr -d '[:space:]' <"${run_directory}/container.id")"
  fi
  if [[ -n "${container_id}" ]] && command -v docker >/dev/null 2>&1 && \
    docker container inspect "${container_id}" >/dev/null 2>&1; then
    capture_to "${output_directory}/container-state.txt" \
      selected_container_inspect "${container_id}"
    capture_to "${output_directory}/container.log" \
      docker logs --timestamps "${container_id}"
  else
    printf 'status=unavailable container_id=%q\n' "${container_id}" \
      >"${output_directory}/container-state.txt"
  fi

  grep -Eai \
    'amdgpu|kfd|xgmi|ring.{0,20}timeout|vm fault|gpu reset|ras|A[E]R|MCE|EDAC|pcieport|IOMMU|RCU stall|blocked for more than|oom|out of memory|watchdog|lockup|panic|ionic|segfault|hardware error' \
    "${output_directory}/saved-boot-full.log" \
    >"${output_directory}/saved-boot-signals.log" || true

  (
    cd -- "${output_directory}"
    find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | \
      xargs -0 sha256sum >SHA256SUMS
  )
  sync_file "${output_directory}/SHA256SUMS"
  sync_file "${output_directory}"
  printf 'Postmortem evidence: %s\n' "${output_directory}"
}

main() {
  [[ $# -ge 1 ]] || { usage >&2; exit 2; }
  local command="$1"
  shift
  case "${command}" in
    run) run_command "$@" ;;
    postmortem) postmortem_command "$@" ;;
    -h | --help | help) usage ;;
    *) die "unknown command: ${command}" ;;
  esac
}

main "$@"
