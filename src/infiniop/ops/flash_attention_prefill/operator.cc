#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/flash_attention_prefill.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/flash_attention_prefill_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateFlashAttentionPrefillDescriptor(
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
    float scale) {

    infiniopTensorDescriptor_t alibi_opt = (alibi_slopes_desc == nullptr) ? nullptr : alibi_slopes_desc;

#define CREATE(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                       \
        return op::flash_attention_prefill::NAMESPACE::Descriptor::create(                            \
            handle,                                                                                  \
            reinterpret_cast<op::flash_attention_prefill::NAMESPACE::Descriptor **>(desc_ptr),       \
            out_desc, q_desc, k_cache_desc, v_cache_desc, block_tables_desc,                         \
            total_kv_lens_desc, cum_seqlens_q_desc,                                                   \
            alibi_opt == nullptr ? std::nullopt : std::make_optional(alibi_opt),                     \
            scale);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopGetFlashAttentionPrefillWorkspaceSize(
    infiniopFlashAttentionPrefillDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                                     \
    case CASE:                                                                                                   \
        *size = reinterpret_cast<op::flash_attention_prefill::NAMESPACE::Descriptor *>(desc)->workspaceSize();   \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopFlashAttentionPrefill(
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
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                                   \
    case CASE:                                                                                       \
        return reinterpret_cast<op::flash_attention_prefill::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, q, k_cache, v_cache, block_tables,                       \
            total_kv_lens, cum_seqlens_q, alibi_slopes, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopDestroyFlashAttentionPrefillDescriptor(
    infiniopFlashAttentionPrefillDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                                    \
    case CASE:                                                                                      \
        delete reinterpret_cast<op::flash_attention_prefill::NAMESPACE::Descriptor *>(desc);        \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

