"""The mlx-lm KV-cache contract that uLLM plugs into (roadmap section 4).

The engine's own caches replace mlx-lm's through the per-layer ``cache``
argument. These checks read the installed mlx-lm so a version change that
moves the plug point fails here, with a clear message, rather than deep inside
an equivalence run (rules 11 and 12).
"""

import inspect

import pytest

pytestmark = pytest.mark.mlx

mx = pytest.importorskip("mlx.core")
base = pytest.importorskip("mlx_lm.models.base")
cache = pytest.importorskip("mlx_lm.models.cache")
qwen3 = pytest.importorskip("mlx_lm.models.qwen3")


def test_pinned_versions_are_installed() -> None:
    import mlx_lm

    assert (mx.__version__, mlx_lm.__version__) == ("0.32.4", "0.32.2")


def test_attention_uses_offset_update_and_fetch_and_fused_sdpa() -> None:
    src = inspect.getsource(qwen3.Attention.__call__)
    for needle in (
        "offset=cache.offset",
        "cache.update_and_fetch(keys, values)",
        "scaled_dot_product_attention",
    ):
        assert needle in src, f"qwen3 attention no longer contains {needle!r}"


def test_the_mask_comes_from_the_first_layer_cache() -> None:
    model_src = inspect.getsource(qwen3.Qwen3Model.__call__)
    assert "create_attention_mask(h, cache[0])" in model_src
    mask_src = inspect.getsource(base.create_attention_mask)
    assert "cache.make_mask(N, return_array=return_array, window_size=window_size)" in mask_src


def test_causal_mask_supports_left_padding() -> None:
    params = inspect.signature(base.create_causal_mask).parameters
    assert list(params) == ["N", "offset", "window_size", "right_padding", "left_padding"]


def test_reference_cache_exposes_the_plug_point() -> None:
    kv = cache.KVCache()
    assert kv.offset == 0
    keys = mx.zeros((1, 8, 3, 128), dtype=mx.bfloat16)  # [B, n_kv_heads, L, head_dim]
    k, v = kv.update_and_fetch(keys, keys)
    assert (kv.offset, k.shape, v.shape) == (3, (1, 8, 3, 128), (1, 8, 3, 128))
    assert hasattr(kv, "make_mask")


def test_mlx_computes_on_the_default_device() -> None:
    a = mx.ones((64, 64))
    assert float((a @ a).sum()) == 64.0 * 64 * 64
