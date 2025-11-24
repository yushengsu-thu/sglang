import torch
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.lora.backend.base_backend import BaseLoRABackend


class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.set_lora: bool = False
        self.lora_backend: BaseLoRABackend = lora_backend
        if hasattr(self.base_layer, "weight"):
            self.weight = self.base_layer.weight

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        pass

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    """
    Vocab parallel embedding layer with support for LoRA (Low-Rank Adaptation).
    
    This implements LoRA for embedding layers by adding low-rank adaptations
    to the embedding weight matrix.
    """

    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.weight = base_layer.weight
        
        # Expose base_layer attributes for compatibility
        self.num_embeddings = base_layer.num_embeddings
        self.embedding_dim = base_layer.embedding_dim
        
        # Get partition info from base_layer (it should already have these)
        if hasattr(base_layer, 'num_embeddings_per_partition'):
            self.num_embeddings_per_partition = base_layer.num_embeddings_per_partition
        elif hasattr(base_layer, 'tp_size'):
            # Calculate based on tp_size from base_layer
            # Ensure divisibility
            assert base_layer.num_embeddings % base_layer.tp_size == 0, \
                f"num_embeddings ({base_layer.num_embeddings}) must be divisible by tp_size ({base_layer.tp_size})"
            self.num_embeddings_per_partition = base_layer.num_embeddings // base_layer.tp_size
        else:
            # Single GPU case - no partitioning
            self.num_embeddings_per_partition = base_layer.num_embeddings
        
        # Store tp_size for use in slice methods
        self.tp_size = getattr(base_layer, 'tp_size', 1)
        
        self.output_offset = torch.tensor(
            [0, base_layer.embedding_dim],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(
        self, 
        base_output: torch.Tensor, 
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply LoRA to embedding output with proper batching and scaling.
        """
        # Get batch info from backend
        if not hasattr(self.lora_backend, 'batch_info') or self.lora_backend.batch_info is None:
            return base_output
        
        batch_info = self.lora_backend.batch_info
        
        # Flatten input for processing
        original_shape = base_output.shape
        input_ids_flat = input_ids.flatten()
        base_output_flat = base_output.view(-1, base_output.shape[-1])
        
        # Initialize output
        lora_output = torch.zeros_like(base_output_flat)
        
        # Get segment information
        seg_indptr = batch_info.seg_indptr  # (num_segments + 1,)
        weight_indices = batch_info.weight_indices  # (num_segments,)
        scalings = batch_info.scalings  # (max_loras_per_batch,)
        
        # Process each segment
        for seg_idx in range(batch_info.num_segments):
            # Get segment boundaries
            start_idx = seg_indptr[seg_idx].item()
            end_idx = seg_indptr[seg_idx + 1].item()
            
            # Get LoRA buffer ID for this segment
            lora_buffer_idx = weight_indices[seg_idx].item()
            
            # Skip if this segment uses base model (lora_buffer_idx == 0)
            if lora_buffer_idx == 0:
                continue
            
            # Get scaling factor for this LoRA
            scaling = scalings[lora_buffer_idx].item()
            
            # Process all tokens in this segment
            for token_idx in range(start_idx, end_idx):
                # Get the actual token ID
                token_id = input_ids_flat[token_idx].item()
                
                # Lookup LoRA A weight for this token
                # A_buffer: (max_loras_per_batch, vocab_size, rank)
                lora_a_vec = self.A_buffer[lora_buffer_idx, token_id, :]  # (rank,)
                
                # Apply LoRA B
                # B_buffer: (max_loras_per_batch, emb_dim, rank)
                lora_contribution = torch.matmul(
                    lora_a_vec, 
                    self.B_buffer[lora_buffer_idx, :, :].T
                )  # (emb_dim,)
                
                # Add scaled LoRA contribution
                lora_output[token_idx] = lora_contribution * scaling
        
        # Add to base output
        output = base_output_flat + lora_output
        
        return output.view(original_shape)

    def forward(self, input_ids: torch.Tensor):
        # Call base embedding layer
        base_output = self.base_layer.forward(input_ids)
        
        if self.set_lora:
            base_output = self.apply_lora(base_output, input_ids)
        
        return base_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        """
        Slice LoRA A weights for tensor parallelism.
        For embeddings, A is indexed by vocab, so we shard along vocab dimension.
        """
        if self.tp_size == 1:
            return A
        
        vocab_size = A.shape[0]
        assert vocab_size % self.tp_size == 0, \
            f"vocab_size ({vocab_size}) must be divisible by tp_size ({self.tp_size})"
        
        shard_size = vocab_size // self.tp_size
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return A[start_idx:end_idx, :].contiguous()

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        """
        Slice LoRA B weights for tensor parallelism.
        For embeddings, B is not sharded as it maps rank to embedding_dim.
        """
        return B


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        shard_size = self.base_layer.output_partition_sizes[0]
        self.output_offset = torch.tensor(
            [
                0,
                shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_)

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        B = B[start_idx:end_idx, :]
        return B


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_gate_up = A_buffer
        self.B_buffer_gate_up = B_buffer

        shard_size = self.base_layer.output_partition_sizes[0]
        self.output_offset = torch.tensor(
            [
                0,
                shard_size,
                2 * shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_gate_up_lora(
            x=x,
            gate_up_lora_a=self.A_buffer_gate_up,
            gate_up_lora_b=self.B_buffer_gate_up,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        gate_size = self.base_layer.output_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return torch.concat(
            (
                B[start_idx:end_idx, :],
                B[gate_size + start_idx : gate_size + end_idx],
            ),
            dim=0,
        )


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        q_proj_shard_size = self.base_layer.q_proj_shard_size
        kv_proj_shard_size = self.base_layer.kv_proj_shard_size
        self.output_offset = torch.tensor(
            [
                0,
                q_proj_shard_size,
                q_proj_shard_size + kv_proj_shard_size,
                q_proj_shard_size + 2 * kv_proj_shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )
        self.output_offset_cpu = self.output_offset.cpu()

        # For computing number of launched blocks
        self.max_qkv_out_dim = max(q_proj_shard_size, kv_proj_shard_size)

    def set_lora_info(
        self,
        A_buffer_qkv: torch.Tensor,
        B_buffer_qkv: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv
        self.B_buffer_qkv = B_buffer_qkv

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_qkv_lora(
            x=x,
            qkv_lora_a=self.A_buffer_qkv,
            qkv_lora_b=self.B_buffer_qkv,
            base_output=base_output,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            max_qkv_out_dim=self.max_qkv_out_dim,
        )
        return lora_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int) -> torch.Tensor:
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas

        q_start_idx = q_proj_shard_size * tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        q_size, k_size, _ = base_layer.output_sizes
        B_q_shard = B[q_start_idx:q_end_idx, :]
        B_k_shard = B[q_size + kv_start_idx : q_size + kv_end_idx, :]
        B_v_shard = B[q_size + k_size + kv_start_idx : q_size + k_size + kv_end_idx, :]

        return torch.concat(
            (
                B_q_shard,
                B_k_shard,
                B_v_shard,
            ),
            dim=0,
        )


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(self, A_buffer: torch.Tensor, B_buffer: torch.Tensor):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        output_size = self.base_layer.output_size
        self.output_offset = torch.tensor(
            [
                0,
                output_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def forward(self, input_: torch.Tensor, skip_all_reduce=False):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if (
            self.base_layer.reduce_results
            and self.base_layer.tp_size > 1
            and not skip_all_reduce
        ):
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        A = A[:, start_idx:end_idx].contiguous()
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        return B


def get_lora_layer(
    layer: nn.Module, lora_backend: BaseLoRABackend
) -> BaseLayerWithLoRA:
    supported_layer_types = {
        # the order matters
        VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")
