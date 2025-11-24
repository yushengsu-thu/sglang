import logging
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch

from sglang.srt.distributed import divide
from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.utils import (
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    LoRAType,
    get_hidden_dim,
    get_normalized_target_modules,
    get_stacked_multiply,
    get_target_module_name,
)
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


class EmptySlot:
    """
    Singleton class to represent an empty slot in the memory pool.
    This is used to improve readability by not using special str as a placeholder.
    """

    __slots__ = ()

    def __repr__(self):
        return "|EMPTY|"

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


EMPTY_SLOT = EmptySlot()


class LoRAMemoryPool:
    """Class for memory pool management of lora modules"""

    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        max_lora_rank: int,
        target_modules: Set[str],
        base_model: torch.nn.Module,
        eviction_policy: str,
    ):
        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = base_hf_config.num_hidden_layers
        self.max_loras_per_batch: int = max_loras_per_batch
        self.dtype: torch.dtype = dtype
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.max_lora_rank: int = max_lora_rank
        self.target_modules: Set[str] = target_modules

        # Initialize eviction policy
        self.eviction_policy = get_eviction_policy(eviction_policy)

        # Both A_buffer and B_buffer maps lora weight names to its buffer space.
        # A_buffer contains num_layer number of row-major tensors with shape
        #   (max_loras_per_batch, stacked_num * max_lora_dim, input_dim)
        # B_buffer contains num_layer number of column-major tensors with shape
        #   (stacked_num, max_loras_per_batch, output_dim, max_lora_dim)
        # Separate buffers for per-layer modules and standalone modules
        self.A_buffer: Dict[str, List[torch.Tensor]] = {}
        self.B_buffer: Dict[str, List[torch.Tensor]] = {}

        # Separate buffers for embedding and lm_head (not per-layer)
        self.embedding_A_buffer: Dict[str, torch.Tensor] = {}
        self.embedding_B_buffer: Dict[str, torch.Tensor] = {}

        # Lora uid -> buffer idx in memory pool
        self.uid_to_buffer_id: Dict[Optional[str], int] = {}

        # Buffer idx -> lora uid in memory pool
        # All uids are initialized as `EmptySlot` for empty buffer slots
        # Here we don't initialize to None since None is a valid uid
        self.buffer_id_to_uid: List[Union[str, None, EmptySlot]] = [
            EMPTY_SLOT
        ] * self.max_loras_per_batch

        self.init_buffers(base_model)

    def can_support(self, config: Union[LoRAConfig, Iterable[LoRAConfig]]) -> bool:
        """
        Check if the memory pool can support the given LoRA adapters.
        """

        def _can_support(config: LoRAConfig) -> bool:
            """
            Check if the memory pool can support a single LoRA adapter.
            """
            if config.r > self.max_lora_rank:
                return False
            target_module_names = get_normalized_target_modules(config.target_modules)
            return target_module_names.issubset(self.target_modules)

        if isinstance(config, LoRAConfig):
            return _can_support(config)
        else:
            return all(_can_support(x) for x in config)

    def get_lora_A_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        input_dim, _ = get_hidden_dim(
            module_name, self.base_hf_config, base_model, layer_idx
        )
        c = get_stacked_multiply(module_name)
        if self.tp_size > 1 and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            input_dim = divide(input_dim, self.tp_size)
        return (
            self.max_loras_per_batch,
            max_lora_dim * c,
            input_dim,
        )

    def get_lora_B_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        _, output_dim = get_hidden_dim(
            module_name, self.base_hf_config, base_model, layer_idx
        )
        if self.tp_size > 1 and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            output_dim = divide(output_dim, self.tp_size)
        return (
            self.max_loras_per_batch,
            output_dim,
            max_lora_dim,
        )

    def init_buffers(self, base_model: torch.nn.Module):
        device = next(base_model.parameters()).device

        def init_buffer(
            buffer: Dict[str, List[torch.Tensor]],
            target_modules: Set[str],
            get_lora_shape_fn: Callable[[str, torch.nn.Module, int, int], Tuple[int]],
        ):
            for module_name in target_modules:
                # Skip embedding modules here - they're handled separately
                if module_name in ["embed_tokens", "lm_head"]:
                    continue
                    
                buffer[module_name] = [
                    torch.empty(
                        get_lora_shape_fn(
                            module_name,
                            base_model,
                            self.max_lora_rank,
                            idx,
                        ),
                        dtype=self.dtype,
                        device=device,
                    )
                    for idx in range(self.num_layer)
                ]

        init_buffer(
            self.A_buffer,
            self.target_modules,
            self.get_lora_A_shape,
        )

        init_buffer(
            self.B_buffer,
            self.target_modules,
            self.get_lora_B_shape,
        )
    
        # Initialize embedding buffers (single, not per-layer)
        for module_name in ["embed_tokens", "lm_head"]:
            if module_name in self.target_modules:
                shape_a = self.get_embedding_lora_A_shape(module_name, base_model)
                shape_b = self.get_embedding_lora_B_shape(module_name, base_model)
                
                self.embedding_A_buffer[module_name] = torch.empty(
                    shape_a, dtype=self.dtype, device=device
                )
                self.embedding_B_buffer[module_name] = torch.empty(
                    shape_b, dtype=self.dtype, device=device
                )


    def get_embedding_lora_A_shape(
        self, module_name: str, 
        base_model: torch.nn.Module) -> Tuple[int]:
        
        """Get shape for embedding LoRA A matrix."""
        # A maps vocab_size -> rank
        if module_name == "embed_tokens":
            module = base_model.model.embed_tokens  # Adjust path based on your model
        elif module_name == "lm_head":
            module = base_model.lm_head
        
        # Handle wrapped LoRA layer - access base_layer if needed
        if hasattr(module, 'base_layer'):
            base_module = module.base_layer
        else:
            base_module = module
        
        vocab_size = (
            base_module.num_embeddings_per_partition 
            if hasattr(base_module, 'num_embeddings_per_partition') 
            else base_module.num_embeddings
        )
        if self.tp_size > 1:
            vocab_size = divide(vocab_size, self.tp_size)
        
        return (self.max_loras_per_batch, vocab_size, self.max_lora_rank)


    def get_embedding_lora_B_shape(
        self, module_name: str, 
        base_model: torch.nn.Module) -> Tuple[int]:
        
        """Get shape for embedding LoRA B matrix."""
        # B maps rank -> embedding_dim
        if module_name == "embed_tokens":
            module = base_model.model.embed_tokens
        elif module_name == "lm_head":
            module = base_model.lm_head
        
        # Handle wrapped LoRA layer - access base_layer if needed
        if hasattr(module, 'base_layer'):
            base_module = module.base_layer
        else:
            base_module = module
        
        embedding_dim = base_module.embedding_dim
        
        return (self.max_loras_per_batch, embedding_dim, self.max_lora_rank)


    def prepare_lora_batch(
        self,
        cur_uids: Set[Optional[str]],
        lora_adapters: Dict[str, LoRAAdapter],
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
        lora_refs: Dict[str, LoRARef],
        embedding_lora_modules: Dict[str, BaseLayerWithLoRA],
    ):
        def get_available_buffer_slot():
            # 1. Prioritize empty slots
            for buffer_id in range(self.max_loras_per_batch):
                if self.buffer_id_to_uid[buffer_id] == EMPTY_SLOT:
                    return buffer_id

            # 2. Memory pool is full, need to evict using policy
            candidates = set()

            for buffer_id in range(self.max_loras_per_batch):
                uid = self.buffer_id_to_uid[buffer_id]

                # Skip if this adapter is needed by current batch
                # TODO (lifuhuang): we might consider supporting pinning base model (uid == None) in the future.
                if uid in cur_uids:
                    continue

                # Skip if this adapter is pinned (base model cannot be pinned, so can be evicted)
                if uid is not None:
                    lora_ref = lora_refs.get(uid)
                    if lora_ref and lora_ref.pinned:
                        continue
                candidates.add(uid)

            if not candidates:
                raise ValueError(
                    "No available buffer slots found. Please ensure the number of active (pinned) loras is less than max_loras_per_batch."
                )

            # Select victim using eviction policy
            victim_uid = self.eviction_policy.select_victim(candidates)

            # Evict the selected victim
            victim_buffer_id = self.uid_to_buffer_id[victim_uid]
            self.uid_to_buffer_id.pop(victim_uid)
            self.eviction_policy.remove(victim_uid)
            self.buffer_id_to_uid[victim_buffer_id] = EMPTY_SLOT
            logger.debug(
                f"Evicting LoRA {victim_uid} from buffer slot {victim_buffer_id}."
            )
            return victim_buffer_id

        # Mark all adapters in current batch as used (for LRU tracking)
        for uid in cur_uids:
            self.eviction_policy.mark_used(uid)

        for uid in cur_uids:
            if uid not in self.uid_to_buffer_id:
                buffer_id = get_available_buffer_slot()
                lora_adapter = lora_adapters.get(uid, None)
                self.load_lora_weight_to_buffer(
                    uid, buffer_id, lora_adapter, lora_modules, embedding_lora_modules
                )
                self.uid_to_buffer_id[uid] = buffer_id
                self.buffer_id_to_uid[buffer_id] = uid

    def load_lora_weight_to_buffer(
        self,
        uid: str,
        buffer_id: int,
        lora_adapter: LoRAAdapter,
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
        embedding_lora_modules: Dict[str, BaseLayerWithLoRA] = None,
    ):
        def load_lora_weight_tensor(
            buffer_view: torch.Tensor, weight: Optional[torch.Tensor]
        ):
            if weight is None:
                # If the particular weight is not present in the adapter, we initialize the buffer to zero
                # to avoid contamination from the residual weight of the evicted adapters.
                buffer_view.zero_()
            else:
                assert (
                    buffer_view.shape == weight.shape
                ), f"LoRA buffer shape {buffer_view.shape} does not match weight shape {weight.shape}."
                buffer_view.copy_(weight)

        # Create short name to module mapping for embedding modules
        embedding_modules_by_short_name = {}
        if embedding_lora_modules:
            for full_name, module in embedding_lora_modules.items():
                short_name = full_name.split(".")[-1]
                embedding_modules_by_short_name[short_name] = module
        
        if uid is None:
            for i in range(self.num_layer):
                for k in self.A_buffer.keys():
                    self.A_buffer[k][i][buffer_id] = 0
            # Also zero out embedding buffers for base model
            for k in self.embedding_A_buffer.keys():
                self.embedding_A_buffer[k][buffer_id] = 0
            for k in self.embedding_B_buffer.keys():
                self.embedding_B_buffer[k][buffer_id] = 0
            return

        assert lora_adapter is not None
        lora_rank = lora_adapter.config.r
        
        # Handle linear layer modules
        for layer_id in range(self.num_layer):
            layer_weights = lora_adapter.layers[layer_id].weights
            temp_A_buffer: Dict[str, Optional[torch.Tensor]] = {
                target_module: None for target_module in self.A_buffer
            }
            temp_B_buffer: Dict[str, Optional[torch.Tensor]] = {
                target_module: None for target_module in self.B_buffer
            }
            for name, weights in layer_weights.items():
                target_module = get_target_module_name(name, self.target_modules)
                if "lora_A" in name:
                    temp_A_buffer[target_module] = weights
                else:
                    temp_B_buffer[target_module] = weights

            if self.tp_size > 1:
                cur_layer_modules = lora_modules[layer_id]
                for module_name, module in cur_layer_modules.items():
                    target_module = get_target_module_name(
                        module_name, self.target_modules
                    )

                    if temp_A_buffer[target_module] is None:
                        # Skip weight slicing if the weight is not present in the adapter
                        continue

                    temp_A_buffer[target_module] = module.slice_lora_a_weights(
                        temp_A_buffer[target_module], self.tp_rank
                    )
                    temp_B_buffer[target_module] = module.slice_lora_b_weights(
                        temp_B_buffer[target_module], self.tp_rank
                    )

            for name, weights in temp_A_buffer.items():
                c = get_stacked_multiply(name)
                target_buffer = self.A_buffer[name][layer_id]
                buffer_view = target_buffer[buffer_id, : lora_rank * c, :]
                load_lora_weight_tensor(buffer_view, weights)

            for name, weights in temp_B_buffer.items():
                target_buffer = self.B_buffer[name][layer_id]
                buffer_view = target_buffer[buffer_id, :, :lora_rank]
                load_lora_weight_tensor(buffer_view, weights)
            
        # Handle embedding modules separately
        if hasattr(lora_adapter, 'embedding_weights'):
            for module_name in ["embed_tokens", "lm_head"]:
                if module_name in self.target_modules:
                    # Find matching keys by checking if module_name is in the key
                    A_weight = None
                    B_weight = None
                    
                    for weight_name, weight in lora_adapter.embedding_weights.items():
                        if module_name in weight_name:
                            if "lora_A" in weight_name:
                                A_weight = weight
                            elif "lora_B" in weight_name:
                                B_weight = weight
                    
                    if A_weight is not None and B_weight is not None:
                        # Slice for tensor parallelism if needed
                        if self.tp_size > 1 and module_name in embedding_modules_by_short_name:
                            module = embedding_modules_by_short_name[module_name]
                            A_weight = module.slice_lora_a_weights(A_weight, self.tp_rank)
                            B_weight = module.slice_lora_b_weights(B_weight, self.tp_rank)
                        
                        # Get buffer references
                        buffer_a = self.embedding_A_buffer[module_name]
                        buffer_b = self.embedding_B_buffer[module_name]
                        
                        # Get expected shapes
                        expected_a_shape = buffer_a[buffer_id, :, :lora_rank].shape  # (vocab_size, rank)
                        expected_b_shape = buffer_b[buffer_id, :, :lora_rank].shape  # (emb_dim, rank)
                        
                        # Auto-transpose if needed (PEFT format is transposed)
                        if A_weight.shape != expected_a_shape:
                            if A_weight.shape == (expected_a_shape[1], expected_a_shape[0]):
                                # Transpose from (rank, vocab_size) to (vocab_size, rank)
                                A_weight = A_weight.T
                            else:
                                raise ValueError(
                                    f"LoRA A weight shape mismatch for {module_name}: "
                                    f"got {A_weight.shape}, expected {expected_a_shape} or transposed"
                                )
                        
                        if B_weight.shape != expected_b_shape:
                            if B_weight.shape == (expected_b_shape[1], expected_b_shape[0]):
                                # Transpose if needed
                                B_weight = B_weight.T
                            else:
                                raise ValueError(
                                    f"LoRA B weight shape mismatch for {module_name}: "
                                    f"got {B_weight.shape}, expected {expected_b_shape} or transposed"
                                )
                        
                        # Copy to buffer
                        buffer_a[buffer_id, :, :lora_rank].copy_(A_weight)
                        buffer_b[buffer_id, :, :lora_rank].copy_(B_weight)


    def get_tensor(
        self, target_module: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:
        if lora_type == LoRAType.LORA_A:
            return self.A_buffer[target_module][layer_id]

        return self.B_buffer[target_module][layer_id]

    def get_embedding_tensor(
        self, module_name: str, lora_type: LoRAType
    ) -> torch.Tensor:
        """
        Get embedding LoRA tensor (for embed_tokens or lm_head).
        Unlike get_tensor(), this doesn't require layer_id since embeddings are not per-layer.
        """
        if lora_type == LoRAType.LORA_A:
            return self.embedding_A_buffer[module_name]

        return self.embedding_B_buffer[module_name]

    def get_buffer_id(self, lora_uid: str):
        return self.uid_to_buffer_id[lora_uid]
