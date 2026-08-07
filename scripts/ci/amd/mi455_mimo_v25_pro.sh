#!/usr/bin/env bash
# Launch XiaomiMiMo/MiMo-V2.5-Pro FP8 on one 4x MI455/gfx1250 host.
#
# MODE=smoke (default) minimizes moving parts for the first health/request test.
# MODE=benchmark enables radix cache and CUDA graphs for cache-hit/perf testing.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/model/MiMo-V2.5-Pro}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8100}"
MODE="${MODE:-smoke}"
MODEL_LOADER_EXTRA_CONFIG="${MODEL_LOADER_EXTRA_CONFIG:-}"
DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-}"
DIAGNOSTICS_RUN_ID="${DIAGNOSTICS_RUN_ID:-mi455-mimo}"
ENABLE_RCCL_DIAGNOSTICS="${ENABLE_RCCL_DIAGNOSTICS:-0}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-0}"
CAPTURE_DIAGNOSTICS_SERVER_LOG="${CAPTURE_DIAGNOSTICS_SERVER_LOG:-1}"
RCCL_DIAGNOSTICS_TO_STDOUT="${RCCL_DIAGNOSTICS_TO_STDOUT:-0}"

validate_boolean() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${name} must be 0 or 1, got ${value}." >&2
    exit 2
  fi
}

validate_boolean ENABLE_RCCL_DIAGNOSTICS "${ENABLE_RCCL_DIAGNOSTICS}"
validate_boolean SKIP_SERVER_WARMUP "${SKIP_SERVER_WARMUP}"
validate_boolean CAPTURE_DIAGNOSTICS_SERVER_LOG "${CAPTURE_DIAGNOSTICS_SERVER_LOG}"
validate_boolean RCCL_DIAGNOSTICS_TO_STDOUT "${RCCL_DIAGNOSTICS_TO_STDOUT}"

PHASE_LOG=""
MANIFEST_FILE=""
SERVER_LOG=""

durable_flush() {
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

phase() {
  local boot_id="unknown"
  local line
  local timestamp

  [[ -n "${PHASE_LOG}" ]] || return 0
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -r /proc/sys/kernel/random/boot_id ]]; then
    boot_id="$(tr -d '[:space:]' </proc/sys/kernel/random/boot_id)"
  fi
  printf -v line 'ts=%s phase=%s pid=%s boot_id=%s' \
    "${timestamp}" "$1" "$$" "${boot_id}"
  if [[ $# -gt 1 ]]; then
    line+=" detail=$2"
  fi
  printf '%s\n' "${line}" >>"${PHASE_LOG}"
  durable_flush "${PHASE_LOG}"
  printf '%s\n' "${line}" >&2
}

on_error() {
  local exit_code="$1"
  local line_number="$2"

  trap - ERR
  set +e
  phase launcher.error "exit=${exit_code} line=${line_number}"
  exit "${exit_code}"
}

if [[ -n "${DIAGNOSTICS_DIR}" ]]; then
  if [[ ! "${DIAGNOSTICS_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "DIAGNOSTICS_RUN_ID must be a safe filename component." >&2
    exit 2
  fi
  if [[ ! -d "${DIAGNOSTICS_DIR}" || ! -w "${DIAGNOSTICS_DIR}" ]]; then
    echo "DIAGNOSTICS_DIR must be a pre-existing writable directory." >&2
    exit 2
  fi
  if [[ "${CAPTURE_DIAGNOSTICS_SERVER_LOG}" == "1" ]] && \
    ! command -v tee >/dev/null 2>&1; then
    echo "tee is required when DIAGNOSTICS_DIR is set." >&2
    exit 2
  fi

  umask 077
  PHASE_LOG="${DIAGNOSTICS_DIR}/${DIAGNOSTICS_RUN_ID}.phases.log"
  MANIFEST_FILE="${DIAGNOSTICS_DIR}/${DIAGNOSTICS_RUN_ID}.manifest.txt"
  diagnostics_files=("${PHASE_LOG}" "${MANIFEST_FILE}")
  if [[ "${CAPTURE_DIAGNOSTICS_SERVER_LOG}" == "1" ]]; then
    SERVER_LOG="${DIAGNOSTICS_DIR}/${DIAGNOSTICS_RUN_ID}.server.log"
    diagnostics_files+=("${SERVER_LOG}")
  fi
  for diagnostics_file in "${diagnostics_files[@]}"; do
    if [[ -e "${diagnostics_file}" || -L "${diagnostics_file}" ]]; then
      echo "Refusing to overwrite diagnostics file: ${diagnostics_file}" >&2
      exit 2
    fi
  done
  for diagnostics_file in "${diagnostics_files[@]}"; do
    : >"${diagnostics_file}"
    chmod 0600 "${diagnostics_file}"
    durable_flush "${diagnostics_file}"
  done
  durable_flush "${DIAGNOSTICS_DIR}"
  if [[ "${CAPTURE_DIAGNOSTICS_SERVER_LOG}" == "1" ]]; then
    exec > >(tee -a "${SERVER_LOG}") 2>&1
  fi

  export PYTHONUNBUFFERED=1
  export PYTHONFAULTHANDLER=1
  if [[ "${ENABLE_RCCL_DIAGNOSTICS}" == "1" ]]; then
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,BOOTSTRAP,ENV,GRAPH,P2P,SHM,NET,RAS}"
    export NCCL_DEBUG_TIMESTAMP_LEVELS="${NCCL_DEBUG_TIMESTAMP_LEVELS:-INFO}"
    if [[ "${RCCL_DIAGNOSTICS_TO_STDOUT}" == "1" ]]; then
      unset NCCL_DEBUG_FILE
    else
      export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-${DIAGNOSTICS_DIR}/rccl.${DIAGNOSTICS_RUN_ID}.%h.%p.log}"
      if [[ "${NCCL_DEBUG_FILE}" != *%p* ]]; then
        echo "NCCL_DEBUG_FILE must contain %p to avoid multi-rank log collisions." >&2
        exit 2
      fi
    fi
  fi
elif [[ "${ENABLE_RCCL_DIAGNOSTICS}" == "1" ]]; then
  echo "ENABLE_RCCL_DIAGNOSTICS=1 requires DIAGNOSTICS_DIR." >&2
  exit 2
fi

if [[ -n "${DIAGNOSTICS_DIR}" ]]; then
  trap 'on_error "$?" "$LINENO"' ERR
fi
phase launcher.start "mode=${MODE} tp=${TP_SIZE}"

if [[ -z "${MODEL_LOADER_EXTRA_CONFIG}" ]]; then
  # The 1.03 TB checkpoint has 34 roughly 30 GB shards. SGLang's default
  # eight-worker buffered loader can retain up to ten shards per TP rank,
  # which creates an excessive host-memory peak when four ranks load in
  # parallel. Prefer bounded, single-threaded loading for first bring-up.
  MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load": false}'
fi

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HSA_COREDUMP_PATTERN="${HSA_COREDUMP_PATTERN:-/dev/null}"

if [[ "${SGLANG_USE_AITER:-0}" != "0" ]]; then
  echo "MiMo-Pro block-FP8 MoE on gfx1250 currently requires SGLANG_USE_AITER=0." >&2
  echo "The pinned AITER gfx1250 grouped-MoE path does not support per_128x128 weights." >&2
  exit 2
fi
if [[ "${ENABLE_CK:-0}" != "0" ]]; then
  echo "gfx1250 correctness mode requires ENABLE_CK=0." >&2
  exit 2
fi

export SGLANG_USE_AITER=0
export AITER_FORCE_A8W4=0
export ENABLE_CK=0

if [[ ! -r "${MODEL_PATH}/config.json" ]]; then
  echo "Missing ${MODEL_PATH}/config.json" >&2
  exit 2
fi
if [[ ! -r "${MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "Missing ${MODEL_PATH}/model.safetensors.index.json (checkpoint incomplete?)" >&2
  exit 2
fi

phase checkpoint_preflight.start
python3 - "${MODEL_PATH}" "${TP_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
tp_size = int(sys.argv[2])
if tp_size <= 0:
    raise SystemExit(f"TP_SIZE must be positive, got {tp_size}.")

with (model_path / "config.json").open() as file:
    config = json.load(file)
with (model_path / "model.safetensors.index.json").open() as file:
    index = json.load(file)

architectures = config.get("architectures") or []
supported_architectures = {"MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM"}
if not architectures or architectures[0] not in supported_architectures:
    raise SystemExit(f"Unexpected model architecture: {architectures!r}.")

text_config = config.get("text_config") or config
quant_config = config.get("quantization_config") or text_config.get(
    "quantization_config", {}
)
block_shape = quant_config.get("weight_block_size")
if block_shape != [128, 128]:
    raise SystemExit(f"Expected 128x128 block-FP8 weights, got {block_shape!r}.")

expected_attention_tp = text_config.get("num_key_value_heads")
if expected_attention_tp is not None and expected_attention_tp % tp_size != 0:
    raise SystemExit(
        f"Fused QKV checkpoint TP {expected_attention_tp} is incompatible with TP {tp_size}."
    )

moe_intermediate_size = text_config.get("moe_intermediate_size")
if moe_intermediate_size is not None:
    if moe_intermediate_size % tp_size != 0:
        raise SystemExit(
            f"MoE intermediate size {moe_intermediate_size} is not divisible by TP {tp_size}."
        )
    if (moe_intermediate_size // tp_size) % block_shape[0] != 0:
        raise SystemExit(
            "The TP-local MoE intermediate dimension is not divisible by the FP8 block size."
        )

weight_map = index.get("weight_map") or {}
shards = sorted(set(weight_map.values()))
if not shards:
    raise SystemExit("Checkpoint index has no weight_map entries.")
missing_shards = [name for name in shards if not (model_path / name).is_file()]
if missing_shards:
    preview = ", ".join(missing_shards[:5])
    suffix = "" if len(missing_shards) <= 5 else ", ..."
    raise SystemExit(
        f"Checkpoint is incomplete: {len(missing_shards)}/{len(shards)} shards missing: "
        f"{preview}{suffix}"
    )

total_size = index.get("metadata", {}).get("total_size", "unknown")
print(
    f"checkpoint preflight passed: architecture={architectures[0]} "
    f"shards={len(shards)} bytes={total_size} tp={tp_size}",
    file=sys.stderr,
)
PY
phase checkpoint_preflight.ok

if [[ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]]; then
  phase gpu_preflight.start
  if [[ ! -r /sys/module/amdgpu/parameters/gpu_recovery ]]; then
    echo "amdgpu is not loaded; run: sudo modprobe amdgpu gpu_recovery=0" >&2
    exit 2
  fi
  gpu_recovery="$(tr -d '[:space:]' </sys/module/amdgpu/parameters/gpu_recovery)"
  if [[ "${gpu_recovery}" != "0" ]]; then
    echo "Expected amdgpu gpu_recovery=0, found ${gpu_recovery}." >&2
    exit 2
  fi
python3 - "${TP_SIZE}" <<'PY'
import importlib.metadata
import sys

import torch

tp_size = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("ROCm is unavailable; verify /dev/kfd and /dev/dri/renderD*.")
if torch.cuda.device_count() < tp_size:
    raise SystemExit(
        f"Need at least {tp_size} GPUs, found {torch.cuda.device_count()}."
    )
for index in range(tp_size):
    arch = torch.cuda.get_device_properties(index).gcnArchName
    if "gfx1250" not in arch:
        raise SystemExit(f"GPU {index} is {arch}, expected gfx1250.")
try:
    triton_version = importlib.metadata.version("triton")
except importlib.metadata.PackageNotFoundError:
    triton_version = "unknown"
print(
    f"gfx1250 preflight passed for {tp_size} GPUs: "
    f"torch={torch.__version__} hip={torch.version.hip} triton={triton_version}",
    file=sys.stderr,
)
PY
  phase gpu_preflight.ok
else
  phase gpu_preflight.skipped
fi

common_args=(
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --tp "${TP_SIZE}"
  --prefill-attention-backend triton
  --decode-attention-backend triton
  --moe-runner-backend triton
  --page-size "${PAGE_SIZE:-64}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}"
  --watchdog-timeout "${WATCHDOG_TIMEOUT:-1200}"
  --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}"
  --port "${PORT}"
)

case "${MODE}" in
  smoke)
    mode_args=(
      --disable-radix-cache
      --disable-cuda-graph
      --disable-custom-all-reduce
      --context-length "${CONTEXT_LENGTH:-16384}"
      --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}"
      --max-running-requests "${MAX_RUNNING_REQUESTS:-4}"
    )
    ;;
  benchmark)
    mode_args=(
      --context-length "${CONTEXT_LENGTH:-131072}"
      --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-131072}"
      --max-running-requests "${MAX_RUNNING_REQUESTS:-64}"
    )
    ;;
  *)
    echo "Unsupported MODE=${MODE}; expected smoke or benchmark." >&2
    exit 2
    ;;
esac

if [[ "${SKIP_SERVER_WARMUP}" == "1" ]]; then
  mode_args+=(--skip-server-warmup)
fi

launch_cmd=(python3)
if [[ -n "${DIAGNOSTICS_DIR}" ]]; then
  launch_cmd+=(-u -X faulthandler)
fi
launch_cmd+=(-m sglang.launch_server "${common_args[@]}" "${mode_args[@]}")

if [[ -n "${MANIFEST_FILE}" ]]; then
  {
    printf 'recorded_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname=%s\n' "$(hostname 2>/dev/null || printf unknown)"
    if [[ -r /proc/sys/kernel/random/boot_id ]]; then
      printf 'boot_id=%s\n' "$(tr -d '[:space:]' </proc/sys/kernel/random/boot_id)"
    fi
    if [[ -r /proc/cmdline ]]; then
      printf 'kernel_cmdline=%q\n' "$(</proc/cmdline)"
    fi
    printf 'uname=%q\n' "$(uname -a 2>/dev/null || printf unknown)"
    printf 'python=%q\n' "$(python3 --version 2>&1 || true)"
    if command -v git >/dev/null 2>&1; then
      printf 'git_head=%s\n' "$(git -c safe.directory="${REPO_ROOT}" -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)"
      printf 'git_status_begin\n'
      git -c safe.directory="${REPO_ROOT}" -C "${REPO_ROOT}" status --porcelain=v1 2>/dev/null || true
      printf 'git_status_end\n'
    fi
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "${BASH_SOURCE[0]}" | sed 's/^/launcher_sha256=/'
    elif command -v shasum >/dev/null 2>&1; then
      shasum -a 256 "${BASH_SOURCE[0]}" | sed 's/^/launcher_sha256=/'
    fi
    printf 'selected_environment_begin\n'
    for variable_name in \
      MODE MODEL_PATH TP_SIZE PORT CUDA_VISIBLE_DEVICES PAGE_SIZE \
      MEM_FRACTION_STATIC WATCHDOG_TIMEOUT MODEL_LOADER_EXTRA_CONFIG \
      CONTEXT_LENGTH CHUNKED_PREFILL_SIZE MAX_RUNNING_REQUESTS \
      SKIP_GPU_PREFLIGHT SKIP_SERVER_WARMUP SGLANG_USE_AITER ENABLE_CK \
      AITER_FORCE_A8W4 PYTHONUNBUFFERED PYTHONFAULTHANDLER \
      PATH PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD DIAGNOSTICS_DIR \
      DIAGNOSTICS_RUN_ID ENABLE_RCCL_DIAGNOSTICS \
      CAPTURE_DIAGNOSTICS_SERVER_LOG RCCL_DIAGNOSTICS_TO_STDOUT \
      TORCH_SHOW_CPP_STACKTRACES HSA_COREDUMP_PATTERN HSA_ENABLE_SDMA \
      HSA_ENABLE_PEER_SDMA HSA_XNACK HSAKMT_DEBUG_LEVEL \
      HIP_LAUNCH_BLOCKING AMD_SERIALIZE_KERNEL NCCL_DEBUG \
      NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE NCCL_ALGO NCCL_PROTO \
      NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE; do
      if declare -p "${variable_name}" >/dev/null 2>&1; then
        printf '%s=%q\n' "${variable_name}" "${!variable_name}"
      else
        printf '%s=<unset>\n' "${variable_name}"
      fi
    done
    printf 'runtime_environment_begin\n'
    while IFS= read -r variable_name; do
      case "${variable_name}" in
        CUDA_* | HIP_* | HSA_* | ROCR_* | ROCM_* | NCCL_* | RCCL_* | \
          TORCH_NCCL_* | SGLANG_* | AITER_* | TRITON_* | MORI_* | OMP_*)
          case "${variable_name}" in
            *KEY* | *TOKEN* | *SECRET* | *PASS* | *CREDENTIAL*)
              printf '%s=<redacted>\n' "${variable_name}"
              ;;
            *) printf '%s=%q\n' "${variable_name}" "${!variable_name}" ;;
          esac
          ;;
      esac
    done < <(compgen -e | LC_ALL=C sort -u)
    printf 'runtime_environment_end\n'
    printf 'python_executable=%q\n' "$(command -v python3 2>/dev/null || printf unknown)"
    printf 'selected_environment_end\n'
    printf 'argv='
    printf ' %q' "${launch_cmd[@]}"
    printf '\n'
  } >"${MANIFEST_FILE}"
  durable_flush "${MANIFEST_FILE}"
fi

echo "Launching MiMo-V2.5-Pro: mode=${MODE} tp=${TP_SIZE} model=${MODEL_PATH}" >&2
phase launch_server.exec "skip_warmup=${SKIP_SERVER_WARMUP}"
exec "${launch_cmd[@]}"
