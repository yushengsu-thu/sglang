#!/usr/bin/env bash
# Launch XiaomiMiMo/MiMo-V2.5-Pro FP8 on one 4x MI455/gfx1250 host.
#
# MODE=smoke (default) minimizes moving parts for the first health/request test.
# MODE=benchmark enables radix cache and CUDA graphs for cache-hit/perf testing.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/model/MiMo-V2.5-Pro}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8100}"
MODE="${MODE:-smoke}"

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

if [[ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]]; then
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
print(f"gfx1250 preflight passed for {tp_size} GPUs", file=sys.stderr)
PY
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
  --model-loader-extra-config '{"enable_multithread_load": true}'
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

echo "Launching MiMo-V2.5-Pro: mode=${MODE} tp=${TP_SIZE} model=${MODEL_PATH}" >&2
exec python3 -m sglang.launch_server "${common_args[@]}" "${mode_args[@]}"
