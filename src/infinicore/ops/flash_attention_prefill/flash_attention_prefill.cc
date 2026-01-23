#include "infinicore/ops/flash_attention_prefill.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<FlashAttentionPrefill::schema> &FlashAttentionPrefill::dispatcher() {
    static common::OpDispatcher<FlashAttentionPrefill::schema> dispatcher_;
    return dispatcher_;
};

void FlashAttentionPrefill::execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                                    Tensor block_tables, Tensor total_kv_lens, Tensor cum_seqlens_q,
                                    std::optional<Tensor> alibi_slopes, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, total_kv_lens, cum_seqlens_q);
    infinicore::context::setDevice(out->device());
    dispatcher().lookup(out->device().getType())(
        out, q, k_cache, v_cache, block_tables, total_kv_lens, cum_seqlens_q, alibi_slopes, scale);
}

Tensor flash_attention_prefill(Tensor q,
                               Tensor k_cache,
                               Tensor v_cache,
                               Tensor block_tables,
                               Tensor total_kv_lens,
                               Tensor cum_seqlens_q,
                               std::optional<Tensor> alibi_slopes,
                               float scale) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    flash_attention_prefill_(out, q, k_cache, v_cache, block_tables, total_kv_lens, cum_seqlens_q, alibi_slopes, scale);
    return out;
}

void flash_attention_prefill_(Tensor out,
                              Tensor q,
                              Tensor k_cache,
                              Tensor v_cache,
                              Tensor block_tables,
                              Tensor total_kv_lens,
                              Tensor cum_seqlens_q,
                              std::optional<Tensor> alibi_slopes,
                              float scale) {
    FlashAttentionPrefill::execute(out, q, k_cache, v_cache, block_tables, total_kv_lens, cum_seqlens_q, alibi_slopes, scale);
}

} // namespace infinicore::op

