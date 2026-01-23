from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def flash_attention(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    cache_lens: Tensor,
    alibi_slopes: Tensor | None = None,
    scale: float = 1.0,
    *,
    out: Tensor | None = None,
):
    if out is None:
        return Tensor(
            _infinicore.flash_attention(
                q._underlying,
                k_cache._underlying,
                v_cache._underlying,
                block_tables._underlying,
                cache_lens._underlying,
                alibi_slopes._underlying if alibi_slopes is not None else None,
                scale,
            )
        )

    _infinicore.flash_attention_(
        out._underlying,
        q._underlying,
        k_cache._underlying,
        v_cache._underlying,
        block_tables._underlying,
        cache_lens._underlying,
        alibi_slopes._underlying if alibi_slopes is not None else None,
        scale,
    )

    return out

