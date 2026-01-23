#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/flash_attention_prefill.hpp"
#include "infinicore/ops/paged_attention_prefill.hpp"
#include <infiniop.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace infinicore::op::flash_attention_prefill_impl::infiniop {

thread_local common::OpCache<size_t, infiniopFlashAttentionPrefillDescriptor_t> caches(
    100, // capacity
    [](infiniopFlashAttentionPrefillDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyFlashAttentionPrefillDescriptor(desc));
            desc = nullptr;
        }
    });

static bool env_bool(const char *name) {
    const char *env = std::getenv(name);
    if (env == nullptr) return false;
    return (std::strcmp(env, "1") == 0) || (std::strcmp(env, "true") == 0);
}

static int env_int(const char *name, int default_v) {
    const char *env = std::getenv(name);
    if (env == nullptr) return default_v;
    const int v = std::atoi(env);
    return (v > 0) ? v : default_v;
}

static Tensor to_fa_paged_kv_view(Tensor kv_cache) {
    // InfiniLM/InfiniCore paged KV layout: [num_blocks, num_kv_heads, block_size, head_dim]
    // FA2 paged-KV convention: [num_blocks, page_block_size, num_kv_heads, head_dim]
    // We can match the FA2 shape by permuting to [0, 2, 1, 3] without data copy.
    return kv_cache->permute({0, 2, 1, 3});
}

void calculate(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
               Tensor block_tables, Tensor total_kv_lens, Tensor cum_seqlens_q,
               std::optional<Tensor> alibi_slopes, float scale) {
    // v0.4: Keep this op safe to enable by default:
    // - If the input configuration doesn't match FA2 paged-KV constraints, fall back to the existing paged prefill.
    // - This allows correctness tests to pass for small block_size (e.g. 8/16) and float32, while we implement
    //   the real FA2-style prefill kernel family (which requires page_block_size % 256 == 0, fp16/bf16).
    const auto dtype = q->dtype();
    const size_t block_size = k_cache->size(2);
    const size_t head_size = q->size(2);
    if (!((dtype == infinicore::DataType::F16) || (dtype == infinicore::DataType::BF16)) ||
        (block_size % 256 != 0) || (head_size != 64 && head_size != 128)) {
        infinicore::op::paged_attention_prefill_(out, q, k_cache, v_cache, block_tables, total_kv_lens, cum_seqlens_q, alibi_slopes, scale);
        return;
    }

    auto k_fa = to_fa_paged_kv_view(k_cache);
    auto v_fa = to_fa_paged_kv_view(v_cache);

    size_t seed = hash_combine(out, q, k_fa, v_fa, block_tables, total_kv_lens, cum_seqlens_q, alibi_slopes, scale);
    // Split-KV prefill (experimental) needs a separate descriptor because workspace size differs.
    // We bake the env toggles into the descriptor cache key to avoid allocating huge workspace
    // when split-kv is not enabled.
    const bool splitkv = env_bool("INFINIOP_FLASH_PREFILL_SPLITKV");
    const int num_splits = env_int("INFINIOP_FLASH_PREFILL_NUM_SPLITS", /*default_v=*/4);
    const bool fa2_materialize_kv = env_bool("INFINIOP_FA2_MATERIALIZE_PAGED_KV");
    // NOTE: infinicore::hash_combine(seed, ...) mutates seed in-place (void return).
    hash_combine(seed, splitkv, static_cast<size_t>(num_splits));
    hash_combine(seed, fa2_materialize_kv);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopFlashAttentionPrefillDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateFlashAttentionPrefillDescriptor(
            context::getInfiniopHandle(device), &desc,
            out->desc(),
            q->desc(),
            k_fa->desc(),
            v_fa->desc(),
            block_tables->desc(),
            total_kv_lens->desc(),
            cum_seqlens_q->desc(),
            alibi_slopes.has_value() ? alibi_slopes.value()->desc() : nullptr,
            scale));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetFlashAttentionPrefillWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    const infiniStatus_t ret = infiniopFlashAttentionPrefill(
        desc,
        workspace->data(),
        workspace_size,
        out->data(),
        q->data(),
        k_fa->data(),
        v_fa->data(),
        block_tables->data(),
        total_kv_lens->data(),
        cum_seqlens_q->data(),
        alibi_slopes.has_value() ? alibi_slopes.value()->data() : nullptr,
        context::getStream());

    // v0.4: flash prefill kernel is not implemented yet; keep a safe fallback so
    // enabling it doesn't break the framework.
    if (ret == INFINI_STATUS_NOT_IMPLEMENTED) {
        static std::atomic<bool> printed{false};
        if (printed.exchange(true) == false) {
            if (const char *env = std::getenv("INFINIOP_FLASH_PREFILL_DEBUG_DISPATCH")) {
                if ((std::strcmp(env, "1") == 0) || (std::strcmp(env, "true") == 0)) {
                    std::fprintf(stderr, "[INFINIOP][flash_attention_prefill] NOT_IMPLEMENTED -> fallback to paged_attention_prefill\n");
                }
            }
        }
        infinicore::op::paged_attention_prefill_(out, q, k_cache, v_cache, block_tables, total_kv_lens, cum_seqlens_q, alibi_slopes, scale);
        return;
    }
    INFINICORE_CHECK_ERROR(ret);
}

static bool registered = []() {
    FlashAttentionPrefill::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::flash_attention_prefill_impl::infiniop
