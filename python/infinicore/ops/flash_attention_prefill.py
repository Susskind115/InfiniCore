from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def flash_attention_prefill(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    total_kv_lens: Tensor,
    cum_seqlens_q: Tensor,
    alibi_slopes: Tensor | None = None,
    scale: float = 1.0,
    *,
    out: Tensor | None = None,
):
    if out is None:
        return Tensor(
            _infinicore.flash_attention_prefill(
                q._underlying,
                k_cache._underlying,
                v_cache._underlying,
                block_tables._underlying,
                total_kv_lens._underlying,
                cum_seqlens_q._underlying,
                alibi_slopes._underlying if alibi_slopes is not None else None,
                scale,
            )
        )

    _infinicore.flash_attention_prefill_(
        out._underlying,
        q._underlying,
        k_cache._underlying,
        v_cache._underlying,
        block_tables._underlying,
        total_kv_lens._underlying,
        cum_seqlens_q._underlying,
        alibi_slopes._underlying if alibi_slopes is not None else None,
        scale,
    )

    return out

