import os
import sys

import torch

import infinicore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import BaseOperatorTest, GenericTestRunner

# Reuse the same test-case generator + torch reference from the paged prefill test.
import paged_attention_prefill as paged_prefill


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("FlashAttentionPrefill")

    def get_test_cases(self):
        # Reuse paged-prefill case generator, but rename descriptions to avoid confusion.
        cases = paged_prefill.parse_test_cases()
        for tc in cases:
            if isinstance(tc.description, str) and tc.description.startswith("PagedAttentionPrefill"):
                tc.description = tc.description.replace("PagedAttentionPrefill", "FlashAttentionPrefill", 1)
        return cases

    def torch_operator(
        self,
        query,
        k_cache,
        v_cache,
        block_tables,
        kv_lens,
        cum_seqlens_q,
        scale=1.0,
    ):
        return paged_prefill.ref_paged_attention_multi_turn(
            query, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, scale
        )

    def infinicore_operator(
        self,
        query,
        k_cache,
        v_cache,
        block_tables,
        kv_lens,
        cum_seqlens_q,
        scale=1.0,
    ):
        # v0.4: flash_attention_prefill currently falls back to paged_attention_prefill
        # when the NVIDIA kernel is NOT_IMPLEMENTED. This test ensures the new op
        # entrypoint is wired correctly and remains functionally consistent.
        out = infinicore.flash_attention_prefill(
            query,
            k_cache,
            v_cache,
            block_tables,
            kv_lens,
            cum_seqlens_q,
            alibi_slopes=None,
            scale=scale,
        )
        infinicore.sync_stream()
        return out


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
