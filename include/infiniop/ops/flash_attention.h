#ifndef __INFINIOP_FLASH_ATTENTION_API_H__
#define __INFINIOP_FLASH_ATTENTION_API_H__

#include "../operator_descriptor.h"

// Define an opaque handle for the Flash Attention descriptor.
typedef struct InfiniopDescriptor *infiniopFlashAttentionDescriptor_t;

/**
 * @brief Creates a descriptor for the Flash Attention (decode-style) operation.
 *
 * External signature aligns with paged_attention:
 * - q: (num_seqs, num_heads, head_size)
 * - out: (num_seqs, num_heads, head_size)
 * - block_tables / cache_lens follow paged-attention semantics (but may be internally converted).
 *
 * Note: This op expects paged KV in FA2-compatible shape:
 *   k_cache/v_cache: (num_blocks, page_block_size, num_kv_heads, head_size)
 * where page_block_size must be a multiple of 256.
 *
 * In InfiniCore C++ wrapper, the original KV cache layout
 *   (num_blocks, num_kv_heads, block_size, head_size)
 * is passed as a permuted view to match the above shape without copying.
 */
__C __export infiniStatus_t infiniopCreateFlashAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t cache_lens_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    float scale);

__C __export infiniStatus_t infiniopGetFlashAttentionWorkspaceSize(
    infiniopFlashAttentionDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopFlashAttention(
    infiniopFlashAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *cache_lens,
    const void *alibi_slopes,
    void *stream);

__C __export infiniStatus_t infiniopDestroyFlashAttentionDescriptor(
    infiniopFlashAttentionDescriptor_t desc);

#endif // __INFINIOP_FLASH_ATTENTION_API_H__

