#ifndef __INFINIOP_FLASH_ATTENTION_PREFILL_API_H__
#define __INFINIOP_FLASH_ATTENTION_PREFILL_API_H__

#include "../operator_descriptor.h"

// Define an opaque handle for the Flash Attention Prefill descriptor.
typedef struct InfiniopDescriptor *infiniopFlashAttentionPrefillDescriptor_t;

/**
 * @brief Creates a descriptor for the Flash Attention (prefill / varlen-style) operation.
 *
 * External signature aligns with paged_attention_prefill:
 * - q/out are packed: [total_q_tokens, num_heads, head_size]
 * - k_cache/v_cache are paged KV cache following FA2 paged-KV convention:
 *   k_cache/v_cache: (num_blocks, page_block_size, num_kv_heads, head_size)
 *
 * Note: In InfiniCore C++ wrapper, the original KV cache layout
 *   (num_blocks, num_kv_heads, block_size, head_size)
 * is passed as a permuted view to match the above shape without copying.
 */
__C __export infiniStatus_t infiniopCreateFlashAttentionPrefillDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionPrefillDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t total_kv_lens_desc,
    infiniopTensorDescriptor_t cum_seqlens_q_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    float scale);

__C __export infiniStatus_t infiniopGetFlashAttentionPrefillWorkspaceSize(
    infiniopFlashAttentionPrefillDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopFlashAttentionPrefill(
    infiniopFlashAttentionPrefillDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *total_kv_lens,
    const void *cum_seqlens_q,
    const void *alibi_slopes,
    void *stream);

__C __export infiniStatus_t infiniopDestroyFlashAttentionPrefillDescriptor(
    infiniopFlashAttentionPrefillDescriptor_t desc);

#endif // __INFINIOP_FLASH_ATTENTION_PREFILL_API_H__

