#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

// FlashAttention (decode-style): q is 1 token per sequence.
// External signature intentionally matches paged_attention_ to minimize framework changes.
class FlashAttention {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>, float);
    static void execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens,
                        std::optional<Tensor> alibi_slopes, float scale);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor flash_attention(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens,
                       std::optional<Tensor> alibi_slopes, float scale);

void flash_attention_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens,
                      std::optional<Tensor> alibi_slopes, float scale);

} // namespace infinicore::op

