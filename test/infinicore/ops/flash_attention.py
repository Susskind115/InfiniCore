import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    TensorInitializer,
)

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# (num_seqs, num_heads, num_kv_heads, head_size, block_size, max_seq_len, use_alibi)
#
# Notes:
# - flash_attention currently targets FA2-style paged KV, requiring block_size % 256 == 0.
# - first build only supports head_size in {64, 128} and dtype in {fp16, bf16}.
_TEST_CASES_DATA = [
    (4, 32, 32, 128, 256, 512, False),
    (8, 64, 8, 128, 256, 1024, False),  # MQA/GQA
    (3, 24, 8, 64, 256, 512, False),  # GQA + head_dim=64
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 3e-2},
    infinicore.bfloat16: {"atol": 0, "rtol": 6e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]


# ==============================================================================
# Reference Implementation
# ==============================================================================


def ref_masked_attention(query, key, value, scale, attn_mask=None):
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    query, key_cache, value_cache, block_tables, cache_lens, alibi_slopes, scale
):
    output = torch.empty_like(query)
    num_query_heads, num_kv_heads = query.shape[1], value_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size, block_size = value_cache.shape[3], value_cache.shape[2]
    num_seqs = query.shape[0]

    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        seq_len = cache_lens[i].item()
        block_table = block_tables[i]

        keys_lst, values_lst = [], []
        for j in range(seq_len):
            block_num = block_table[j // block_size].item()
            block_off = j % block_size
            k = key_cache[block_num, :, block_off, :]
            v = value_cache[block_num, :, block_off, :]
            keys_lst.append(k)
            values_lst.append(v)

        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            pos = torch.arange(seq_len, device=query.device).int()
            alibi_bias = (pos - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        output[i] = out.view(num_query_heads, head_size)

    return output


def parse_test_cases():
    test_cases = []
    index_dtypes = [
        (torch.int64, infinicore.int64),
        (torch.int32, infinicore.int32),
    ]
    for (
        num_seqs,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_seq_len,
        use_alibi,
    ) in _TEST_CASES_DATA:
        scale = 1.0 / (head_size**0.5)
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        num_blocks = num_seqs * max_blocks_per_seq

        cache_lens_i64 = torch.randint(1, max_seq_len, (num_seqs,), dtype=torch.int64)
        block_tables_i64 = torch.arange(0, num_seqs * max_blocks_per_seq, dtype=torch.int64).view(
            num_seqs, max_blocks_per_seq
        )

        q_shape = (num_seqs, num_heads, head_size)
        k_cache_shape = (num_blocks, num_kv_heads, block_size, head_size)
        v_cache_shape = (num_blocks, num_kv_heads, block_size, head_size)

        for torch_index_dtype, infinicore_index_dtype in index_dtypes:
            cache_lens_torch = cache_lens_i64.to(torch_index_dtype)
            block_tables = block_tables_i64.to(torch_index_dtype)

            for dtype in _TENSOR_DTYPES:
                tolerance = _TOLERANCE_MAP[dtype]

                q_spec = TensorSpec.from_tensor(q_shape, None, dtype)
                k_cache_spec = TensorSpec.from_tensor(k_cache_shape, None, dtype)
                v_cache_spec = TensorSpec.from_tensor(v_cache_shape, None, dtype)
                block_tables_spec = TensorSpec.from_tensor(
                    block_tables.shape,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=block_tables,
                    dtype=infinicore_index_dtype,
                )
                cache_lens_spec = TensorSpec.from_tensor(
                    cache_lens_torch.shape,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=cache_lens_torch,
                    dtype=infinicore_index_dtype,
                )

                test_cases.append(
                    TestCase(
                        inputs=[q_spec, k_cache_spec, v_cache_spec, block_tables_spec, cache_lens_spec],
                        kwargs={"alibi_slopes": None if not use_alibi else None, "scale": scale},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tolerance,
                        description=f"FlashAttention (index={torch_index_dtype})",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("FlashAttention")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return ref_single_query_cached_kv_attention(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        out = infinicore.flash_attention(*args, **kwargs)
        infinicore.sync_stream()
        return out


def main():
    # Default to NVIDIA when available to avoid accidentally running on CPU,
    # because flash_attention is currently implemented for NVIDIA only.
    if not any(
        flag in sys.argv[1:]
        for flag in (
            "--cpu",
            "--nvidia",
            "--cambricon",
            "--ascend",
            "--metax",
            "--moore",
            "--iluvatar",
            "--kunlun",
            "--hygon",
            "--qy",
        )
    ):
        if torch.cuda.is_available():
            sys.argv.append("--nvidia")
        else:
            print(
                "FlashAttention currently supports NVIDIA only; please run with --nvidia on a CUDA machine."
            )
            sys.exit(0)

    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
