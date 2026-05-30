"""Two-stream attention LoRA forward implementations (O7 + O8).

These are monkey-patched onto :class:`QKVParallelLinearWithLoRA` and
:class:`RowParallelLinearWithLoRA` by
:func:`sglang.srt.lora.trtllm_moe.install_two_stream_overrides` when
``SGLANG_LORA_TWO_STREAM=1``. The saved-original forward methods are
preserved and called for batches where two-stream isn't active.

LoRA-B (expand) scheduling is controlled by ``SGLANG_LORA_B_OVERLAP``
(:class:`LoRABOverlapMode`):

  * OFF       — LoRA-A shrink overlaps the base gemm; LoRA-B runs on the main
                stream after the rejoin (it fuses its add into base_output, so
                it must wait for the base gemm). LoRA-B is *not* overlapped.
  * DECOUPLE  — LoRA-A shrink + LoRA-B expand both run on the side stream into a
                fresh temp buffer; the cheap add to base_output happens on the
                main stream after both finish. LoRA-B now overlaps the base gemm.
  * GREEN_CTX — like DECOUPLE, but the LoRA work runs on a green-context stream
                with a reserved SM partition and the base DeepGEMM is capped to
                the complementary SM count, forcing true co-execution.
"""
import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.environ import LoRABOverlapMode
from sglang.srt.lora.trtllm_moe import (
    get_lora_b_overlap_mode,
    get_lora_green_base_sms,
    get_lora_green_streams,
    get_lora_side_stream,
    get_original_qkv_forward,
    get_original_row_forward,
    is_two_stream_active,
)


def _resolve_lora_base_streams(mode: LoRABOverlapMode):
    """Return (lora_stream, base_stream). base_stream is None ⇒ base runs on the
    main stream (DECOUPLE) instead of a dedicated green-ctx partition (GREEN_CTX)."""
    if mode == LoRABOverlapMode.GREEN_CTX:
        green = get_lora_green_streams()
        if green is not None:
            return green  # (lora_stream, base_stream)
    return get_lora_side_stream(), None


def _run_base_gemm(layer, x, bias, base_stream):
    """Run the base linear. On a green-ctx base_stream, cap DeepGEMM to that
    partition's SM count so it leaves room for the concurrent LoRA stream."""
    apply = layer.base_layer.quant_method.apply
    if base_stream is None:
        return apply(layer.base_layer, x, bias)
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
        configure_deep_gemm_num_sms,
    )

    base_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(base_stream):
        with configure_deep_gemm_num_sms(get_lora_green_base_sms()):
            return apply(layer.base_layer, x, bias)


def qkv_proj_lora_forward(self, input_: torch.Tensor):
    """O7 — side-stream LoRA shrink (+ optionally expand) ‖ base qkv_proj GEMM."""
    if not self.set_lora or not is_two_stream_active(input_):
        return get_original_qkv_forward()(self, input_)

    from sglang.srt.lora.triton_ops import qkv_lora_b_fwd, sgemm_lora_a_fwd

    bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
    # sgemm_info is host-side (LoRABatchInfo); compute once, share both calls.
    sgemm_info = self.lora_backend._sgemm_info()
    mode = get_lora_b_overlap_mode()
    main = torch.cuda.current_stream()

    if mode == LoRABOverlapMode.OFF:
        # LoRA-A shrink on side ‖ base; LoRA-B fused-add on main after rejoin.
        side_stream = get_lora_side_stream()
        side_stream.wait_stream(main)
        with torch.cuda.stream(side_stream):
            shrink_intermediate = sgemm_lora_a_fwd(
                input_, self.A_buffer_qkv, sgemm_info, stack_num=3
            )
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )
        main.wait_stream(side_stream)
        output_parallel = qkv_lora_b_fwd(
            shrink_intermediate,
            self.B_buffer_qkv,
            sgemm_info,
            self.output_offset,
            self.max_qkv_out_dim,
            output_parallel,
            n_slices=3,
        )
    else:
        # DECOUPLE / GREEN_CTX: shrink + expand into a fresh delta on the lora
        # stream, concurrent with the base gemm; cheap add on main after rejoin.
        lora_stream, base_stream = _resolve_lora_base_streams(mode)
        lora_stream.wait_stream(main)
        with torch.cuda.stream(lora_stream):
            shrink_intermediate = sgemm_lora_a_fwd(
                input_, self.A_buffer_qkv, sgemm_info, stack_num=3
            )
            lora_delta = qkv_lora_b_fwd(
                shrink_intermediate,
                self.B_buffer_qkv,
                sgemm_info,
                self.output_offset,
                self.max_qkv_out_dim,
                None,  # fresh buffer — no dependence on base_output
                n_slices=3,
            )
        output_parallel = _run_base_gemm(self, input_, bias, base_stream)
        main.wait_stream(lora_stream)
        if base_stream is not None:
            main.wait_stream(base_stream)
        output_parallel = output_parallel.add_(lora_delta)

    if self.base_layer.gather_output:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output, output_bias


def row_parallel_lora_forward(
    self, input_: torch.Tensor, skip_all_reduce: bool = False, forward_batch=None
):
    """O8 — side-stream LoRA shrink (+ optionally expand) ‖ base o_proj GEMM.

    The TP all-reduce path (``should_reduce``) keeps the OFF schedule: LoRA-A is
    overlapped, but LoRA-B must follow the all-reduce of both base output and the
    shrink intermediate, so it cannot be hoisted ahead of the base gemm. The
    no-reduce path uses the DECOUPLE/GREEN_CTX overlap.
    """
    if self.base_layer.input_is_parallel:
        input_parallel = input_
    else:
        tp_rank = get_tensor_model_parallel_rank()
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=self.base_layer.tp_size
        )
        input_parallel = splitted_input[tp_rank].contiguous()

    if not self.set_lora or not is_two_stream_active(input_parallel):
        return get_original_row_forward()(self, input_, skip_all_reduce, forward_batch)

    bias_ = (
        None
        if (self.base_layer.tp_rank > 0 or self.base_layer.skip_bias_add)
        else self.base_layer.bias
    )
    should_reduce = (
        self.base_layer.reduce_results
        and self.base_layer.tp_size > 1
        and not skip_all_reduce
    )
    mode = get_lora_b_overlap_mode()
    main = torch.cuda.current_stream()

    if should_reduce or mode == LoRABOverlapMode.OFF:
        # LoRA-A shrink on side ‖ base; LoRA-B after rejoin (and after all-reduce).
        side_stream = get_lora_side_stream()
        side_stream.wait_stream(main)
        with torch.cuda.stream(side_stream):
            lora_a_output = self.lora_backend.run_lora_a_sgemm(
                input_parallel, self.A_buffer
            )
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel, bias=bias_
        )
        main.wait_stream(side_stream)
        if should_reduce:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
            lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)
            output_ = self.lora_backend.run_lora_b_sgemm(
                x=lora_a_output,
                weights=self.B_buffer,
                output_offset=self.output_offset,
                output_offset_cpu=self.output_offset_cpu,
                base_output=output_,
            )
        else:
            output_parallel = self.lora_backend.run_lora_b_sgemm(
                x=lora_a_output,
                weights=self.B_buffer,
                output_offset=self.output_offset,
                output_offset_cpu=self.output_offset_cpu,
                base_output=output_parallel,
            )
            output_ = output_parallel
    else:
        # No reduce: shrink + expand into fresh delta on the lora stream ‖ base.
        lora_stream, base_stream = _resolve_lora_base_streams(mode)
        lora_stream.wait_stream(main)
        with torch.cuda.stream(lora_stream):
            lora_a_output = self.lora_backend.run_lora_a_sgemm(
                input_parallel, self.A_buffer
            )
            lora_delta = self.lora_backend.run_lora_b_sgemm(
                x=lora_a_output,
                weights=self.B_buffer,
                output_offset=self.output_offset,
                output_offset_cpu=self.output_offset_cpu,
                base_output=None,
            )
        output_parallel = _run_base_gemm(self, input_parallel, bias_, base_stream)
        main.wait_stream(lora_stream)
        if base_stream is not None:
            main.wait_stream(base_stream)
        output_ = output_parallel.add_(lora_delta)

    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output_, output_bias
