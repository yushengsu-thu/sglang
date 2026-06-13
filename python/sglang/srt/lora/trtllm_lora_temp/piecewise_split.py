"""opt8: piecewise-CUDA-graph split op for the experimental_sgl_trtllm MoE-LoRA dispatch.

The MoE-LoRA dispatch subtree is host-side Python (per-batch LoRAInfo build, Triton
config lookup via torch.cuda.get_device_name(), two-stream logic) that torch._dynamo
cannot trace — each is a hard `Unsupported`, not a recoverable graph break, because
the piecewise compile runs fullgraph. Mirror the RadixAttention escape hatch
(radix_attention.py: unified_attention_with_output): register ONE opaque custom op
that is also a piecewise SPLIT op — the surrounding graph pieces get captured, and
the MoE-LoRA block runs eagerly between them with the REAL per-batch LoRA state.
This also guarantees correctness: capture-time `lora_ids=[None]*bs` is never baked
into a graph, because everything LoRA-dependent re-executes eagerly per call.

Non-tensor state (the layer object, per-batch LoRA info) is fetched inside the op
via a layer registry + the piecewise forward context, exactly like attention does.
"""

from typing import Dict, Optional

import torch

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.utils.custom_op import register_custom_op

_MOE_LORA_LAYERS: Dict[int, object] = {}


def register_moe_lora_layer(layer_id: int, layer) -> None:
    _MOE_LORA_LAYERS[layer_id] = layer


@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_moe_lora_with_output(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
    packed_topk_ids: Optional[torch.Tensor] = None,
) -> None:
    from sglang.srt.layers.moe.topk import (
        StandardTopKOutput,
        StandardTopKOutputPacked,
    )

    layer = _MOE_LORA_LAYERS[layer_id]
    context = get_forward_context()
    real_num_tokens = context.forward_batch.num_token_non_padded_cpu

    hs = hidden_states[:real_num_tokens]
    if packed_topk_ids is not None:
        topk_output = StandardTopKOutputPacked(
            topk_weights[:real_num_tokens],
            topk_ids[:real_num_tokens],
            router_logits[:real_num_tokens],
            packed_topk_ids[:real_num_tokens],
        )
    else:
        topk_output = StandardTopKOutput(
            topk_weights[:real_num_tokens],
            topk_ids[:real_num_tokens],
            router_logits[:real_num_tokens],
        )

    lora_info = layer._get_lora_info()
    out = layer._forward_with_lora(hs, topk_output, lora_info)
    output[:real_num_tokens].copy_(out)


_EMBEDDING_LORA_LAYER = []


def register_embedding_lora_layer(layer) -> None:
    if not _EMBEDDING_LORA_LAYER:
        _EMBEDDING_LORA_LAYER.append(layer)
    else:
        _EMBEDDING_LORA_LAYER[0] = layer


@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_embedding_lora_with_output(
    base_output: torch.Tensor,
    input_ids: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """opt8 step7f: the embedding LoRA kernel was the only in-graph consumer of
    lora_backend.batch_info — its grid reads host ints (max_len/bs) that dynamo
    guards by exact value, and decode/extend swap the batch_info OBJECT per
    forward, overflowing the recompile cache (replay -> 'PCG capture stream is
    not set'). Run it eagerly between graph pieces instead, same as the MoE
    dispatch above."""
    layer = _EMBEDDING_LORA_LAYER[0]
    context = get_forward_context()
    real = context.forward_batch.num_token_non_padded_cpu
    out = layer.apply_lora(
        base_output[:real], input_ids[:real], layer.lora_backend.batch_info
    )
    output[:real].copy_(out)
    if output.shape[0] > real:
        output[real:].copy_(base_output[real:])
