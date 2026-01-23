#include "infinicore/ops/flash_attention.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<FlashAttention::schema> &FlashAttention::dispatcher() {
    static common::OpDispatcher<FlashAttention::schema> dispatcher_;
    return dispatcher_;
};

void FlashAttention::execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens,
                             std::optional<Tensor> alibi_slopes, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, cache_lens);
    infinicore::context::setDevice(out->device());
    dispatcher().lookup(out->device().getType())(out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes, scale);
}

Tensor flash_attention(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens,
                       std::optional<Tensor> alibi_slopes, float scale) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    flash_attention_(out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes, scale);
    return out;
}

void flash_attention_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens,
                      std::optional<Tensor> alibi_slopes, float scale) {
    FlashAttention::execute(out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes, scale);
}

} // namespace infinicore::op

