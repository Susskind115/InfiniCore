#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/flash_attention.hpp"
#include <infiniop.h>

#include <cstdlib>
#include <cstring>

namespace infinicore::op::flash_attention_impl::infiniop {

thread_local common::OpCache<size_t, infiniopFlashAttentionDescriptor_t> caches(
    100, // capacity
    [](infiniopFlashAttentionDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyFlashAttentionDescriptor(desc));
            desc = nullptr;
        }
    });

thread_local common::OpCache<size_t, std::shared_ptr<Memory>> workspace_caches(
    100, // capacity
    [](std::shared_ptr<Memory> &m) {
        m.reset();
    });

static Tensor to_fa_paged_kv_view(Tensor kv_cache) {
    // InfiniLM/InfiniCore current paged KV layout: [num_blocks, num_kv_heads, block_size, head_dim]
    // FA2 paged-KV convention: [num_blocks, page_block_size, num_kv_heads, head_dim]
    // We can match the FA2 shape by permuting to [0, 2, 1, 3] without data copy.
    return kv_cache->permute({0, 2, 1, 3});
}

void calculate(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens,
               std::optional<Tensor> alibi_slopes, float scale) {
    auto k_fa = to_fa_paged_kv_view(k_cache);
    auto v_fa = to_fa_paged_kv_view(v_cache);

    size_t seed = hash_combine(out, q, k_fa, v_fa, block_tables, cache_lens, alibi_slopes, scale);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopFlashAttentionDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateFlashAttentionDescriptor(
            context::getInfiniopHandle(device), &desc,
            out->desc(), q->desc(), k_fa->desc(), v_fa->desc(), block_tables->desc(), cache_lens->desc(),
            alibi_slopes.has_value() ? alibi_slopes.value()->desc() : nullptr,
            scale));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // Split-KV decode needs workspace. Since hd128 decode defaults to split-kv, we also need to allocate
    // workspace when INFINIOP_FLASH_DECODE_SPLITKV is unset (head_dim==128).
    bool need_workspace = false;
    if (const char *env = std::getenv("INFINIOP_FLASH_DECODE_SPLITKV")) {
        need_workspace = (std::strcmp(env, "auto") == 0) ||
                         (std::strcmp(env, "1") == 0) || (std::strcmp(env, "true") == 0);
    } else {
        // q shape is typically [total_q_tokens, num_heads, head_dim] for this op.
        auto q_shape = q->shape();
        const size_t head_dim = q_shape.empty() ? 0 : q_shape.back();
        need_workspace = (head_dim == 128);
    }
    void *workspace_ptr = nullptr;
    size_t workspace_size = 0;
    std::shared_ptr<Memory> workspace;
    if (need_workspace) {
        INFINICORE_CHECK_ERROR(infiniopGetFlashAttentionWorkspaceSize(desc, &workspace_size));
        auto ws_opt = workspace_caches.getCache(device).get(seed);
        if (!ws_opt || (*ws_opt) == nullptr || (*ws_opt)->size() < workspace_size) {
            workspace = context::allocateMemory(workspace_size);
            workspace_caches.getCache(device).put(seed, workspace);
        } else {
            workspace = *ws_opt;
        }
        workspace_ptr = workspace->data();
    }

    INFINICORE_CHECK_ERROR(infiniopFlashAttention(
        desc, workspace_ptr, workspace_size,
        out->data(), q->data(), k_fa->data(), v_fa->data(), block_tables->data(), cache_lens->data(),
        alibi_slopes.has_value() ? alibi_slopes.value()->data() : nullptr,
        context::getStream()));
}

static bool registered = []() {
    FlashAttention::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::flash_attention_impl::infiniop
