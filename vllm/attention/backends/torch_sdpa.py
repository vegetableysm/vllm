""" Attention layer with torch scaled_dot_product_attention
    and PagedAttention."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vineyard.llm import KVCache
from vineyard.llm import KVTensor
from vineyard.llm.config import FileCacheConfig
from vineyard.llm.config import VineyardCacheConfig

class TorchSDPABackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["TorchSDPABackendImpl"]:
        return TorchSDPABackendImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "TorchSDPAMetadata":
        return TorchSDPAMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class TorchSDPAMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for TorchSDPABackend.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    slot_mapping: torch.Tensor
    prompt_lens: Optional[List[int]]
    prompt_lens_tensor: Optional[torch.Tensor]
    num_prompt_tokens: int
    num_generation_tokens: int

    max_subquery_len: Optional[int] = None
    max_prompt_len: Optional[int] = None
    subquery_start_loc: Optional[torch.Tensor] = None
    seq_start_loc: Optional[torch.Tensor] = None
    use_cuda_graph: bool = False

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[torch.Tensor]] = None
        self.vineyard_kv_cache: Optional[torch.Tensor] = None
        self.vineyard_kv_cache_size = 0
        self.vineyard_cache_layer = 0
        self.vineyard_cache_update_size = 0
        self.vineyard_k_cache_update: Optional[torch.Tensor] = None
        self.vineyard_v_cache_update: Optional[torch.Tensor] = None
        self.llm_cache_manager = None


class TorchSDPABackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            assert len(alibi_slopes) == num_heads
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: TorchSDPAMetadata,
        kv_scale: float,
    ) -> torch.Tensor:
        """Forward pass with torch SDPA and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        print("key:", key)

        attn_metadata.vineyard_cache_update_size = num_tokens
        if attn_metadata.vineyard_k_cache_update is None:
            attn_metadata.vineyard_k_cache_update = key.reshape(1, num_tokens, -1).movedim(0, 1)
            attn_metadata.vineyard_v_cache_update = value.reshape(1, num_tokens, -1).movedim(0, 1)
        else:
            attn_metadata.vineyard_k_cache_update = torch.cat((attn_metadata.vineyard_k_cache_update, key.reshape(1, num_tokens, -1).movedim(0, 1)), dim=1)
            attn_metadata.vineyard_v_cache_update = torch.cat((attn_metadata.vineyard_v_cache_update, value.reshape(1, num_tokens, -1).movedim(0, 1)), dim=1)
        print("layer:", attn_metadata.vineyard_cache_layer,"write k:", attn_metadata.vineyard_k_cache_update)

        if attn_metadata.vineyard_kv_cache is not None:
            vineyard_key_cache = torch.empty(0, 4096, dtype=torch.bfloat16)
            vineyard_value_cache = torch.empty(0, 4096, dtype=torch.bfloat16)
            for i in range(attn_metadata.vineyard_kv_cache_size):
                vineyard_key_cache = torch.cat([vineyard_key_cache, attn_metadata.vineyard_kv_cache[i][attn_metadata.vineyard_cache_layer][0].reshape(1, -1)], dim=0)
                vineyard_value_cache = torch.cat([vineyard_value_cache, attn_metadata.vineyard_kv_cache[i][attn_metadata.vineyard_cache_layer][1].reshape(1, -1)], dim=0)
            attn_metadata.vineyard_cache_layer += 1
            key = torch.cat([vineyard_key_cache, key], dim=0)
            value = torch.cat([vineyard_value_cache, value], dim=0)

        print("layer:", attn_metadata.vineyard_cache_layer,"read k:", key)
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                attn_metadata.kv_cache_dtype,
                                                kv_scale)

        if attn_metadata.is_prompt:
            if (kv_cache is None or attn_metadata.block_tables.numel() == 0):
                if self.num_kv_heads != self.num_heads:
                    key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
                    value = value.repeat_interleave(self.num_queries_per_kv,
                                                    dim=1)

                if attn_metadata.attn_bias is None:
                    if self.alibi_slopes is not None:
                        att_masks = _make_alibi_bias(
                            self.alibi_slopes, query.dtype,
                            attn_metadata.prompt_lens)  # type: ignore
                    elif self.sliding_window is not None:
                        att_masks = _make_sliding_window_bias(
                            attn_metadata.prompt_lens, self.sliding_window,
                            query.dtype)  # type: ignore
                    else:
                        att_masks = [None] * len(attn_metadata.prompt_lens)
                    attn_metadata.attn_bias = att_masks

                query = query.movedim(0, query.dim() - 2)
                key = key.movedim(0, key.dim() - 2)
                value = value.movedim(0, value.dim() - 2)

                start = 0
                output = torch.empty(
                    (num_tokens, self.num_heads, self.head_size),
                    dtype=query.dtype)
                for prompt_len, mask in zip(attn_metadata.prompt_lens,
                                            attn_metadata.attn_bias):
                    end = start + prompt_len
                    sub_out = custom_scaled_dot_product_attention(
                        query[:, start:end, :],
                        key[:, start:end + attn_metadata.vineyard_kv_cache_size, :],
                        value[:, start:end + attn_metadata.vineyard_kv_cache_size, :],
                        # key[:, :, :],
                        # value[:, :, :],
                        attn_mask=mask,
                        dropout_p=0.0,
                        is_causal=not self.need_mask,
                        scale=self.scale).movedim(query.dim() - 2, 0)
                    output[start:end, :, :] = sub_out
                    start = end
            else:
                # prefix-enabled attention
                raise RuntimeError(
                    "Torch SDPA backend doesn't support prefix decoding.")

        else:
            # Decoding run.
            output = PagedAttention.forward_decode(
                query,
                key_cache,
                value_cache,
                attn_metadata.block_tables,
                attn_metadata.context_lens,
                attn_metadata.max_context_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
    prompt_lens: List[int],
) -> List[torch.Tensor]:
    attn_biases = []
    for prompt_len in prompt_lens:
        bias = torch.arange(prompt_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(prompt_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        bias = bias[None, :] - bias[:, None]

        num_heads = alibi_slopes.shape[0]
        bias = bias[None, :].expand(num_heads, prompt_len, prompt_len)
        bias.mul_(alibi_slopes[:, None, None])
        inf_mask = torch.empty(
            (1, prompt_len, prompt_len),
            dtype=bias.dtype).fill_(-torch.inf).triu_(diagonal=1)
        attn_biases.append((bias + inf_mask).to(dtype))

    return attn_biases


def _make_sliding_window_bias(
    prompt_lens: List[int],
    window_size: Optional[int],
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    attn_biases = []
    for prompt_len in prompt_lens:
        tensor = torch.full(
            (1, prompt_len, prompt_len),
            dtype=dtype,
            fill_value=1,
        )
        shift = 0
        mask = torch.tril(tensor, diagonal=shift).to(dtype)  # type: ignore
        if window_size is not None:
            mask = torch.triu(mask, diagonal=shift - window_size + 1)
        mask = torch.log(mask)
        attn_biases.append(mask.to(dtype))

    return attn_biases

def custom_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value