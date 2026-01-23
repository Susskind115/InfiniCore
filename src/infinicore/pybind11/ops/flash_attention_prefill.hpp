#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/flash_attention_prefill.hpp"

namespace infinicore::ops {

inline Tensor py_flash_attention_prefill(Tensor q,
                                         Tensor k_cache,
                                         Tensor v_cache,
                                         Tensor block_tables,
                                         Tensor total_kv_lens,
                                         Tensor cum_seqlens_q,
                                         pybind11::object alibi_slopes,
                                         float scale) {
    std::optional<Tensor> alibi_slopes_tensor;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = pybind11::cast<Tensor>(alibi_slopes);
    }
    return op::flash_attention_prefill(q, k_cache, v_cache, block_tables, total_kv_lens, cum_seqlens_q, alibi_slopes_tensor, scale);
}

inline void py_flash_attention_prefill_(Tensor out,
                                        Tensor q,
                                        Tensor k_cache,
                                        Tensor v_cache,
                                        Tensor block_tables,
                                        Tensor total_kv_lens,
                                        Tensor cum_seqlens_q,
                                        pybind11::object alibi_slopes,
                                        float scale) {
    std::optional<Tensor> alibi_slopes_tensor;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = pybind11::cast<Tensor>(alibi_slopes);
    }
    op::flash_attention_prefill_(out, q, k_cache, v_cache, block_tables, total_kv_lens, cum_seqlens_q, alibi_slopes_tensor, scale);
}

inline void bind_flash_attention_prefill(pybind11::module &m) {
    m.def("flash_attention_prefill",
          &ops::py_flash_attention_prefill,
          pybind11::arg("q"),
          pybind11::arg("k_cache"),
          pybind11::arg("v_cache"),
          pybind11::arg("block_tables"),
          pybind11::arg("total_kv_lens"),
          pybind11::arg("cum_seqlens_q"),
          pybind11::arg("alibi_slopes") = pybind11::none(),
          pybind11::arg("scale") = 1.0f);

    m.def("flash_attention_prefill_",
          &ops::py_flash_attention_prefill_,
          pybind11::arg("out"),
          pybind11::arg("q"),
          pybind11::arg("k_cache"),
          pybind11::arg("v_cache"),
          pybind11::arg("block_tables"),
          pybind11::arg("total_kv_lens"),
          pybind11::arg("cum_seqlens_q"),
          pybind11::arg("alibi_slopes") = pybind11::none(),
          pybind11::arg("scale") = 1.0f);
}

} // namespace infinicore::ops
