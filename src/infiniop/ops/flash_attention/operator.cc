#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/flash_attention.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/flash_attention_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateFlashAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t cache_lens_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    float scale) {

#define CREATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        return op::flash_attention::NAMESPACE::Descriptor::create(                    \
            handle,                                                                   \
            reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, q_desc, k_cache_desc, v_cache_desc, block_tables_desc, cache_lens_desc, \
            alibi_slopes_desc == nullptr ? std::nullopt : std::make_optional(alibi_slopes_desc), \
            scale);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopGetFlashAttentionWorkspaceSize(
    infiniopFlashAttentionDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                        \
        *size = reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopFlashAttention(
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
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                                 \
        return reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopDestroyFlashAttentionDescriptor(
    infiniopFlashAttentionDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                    \
    case CASE:                                                                      \
        delete reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

