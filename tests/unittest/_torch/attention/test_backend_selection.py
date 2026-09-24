# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for attention backend selection and fallback diagnostics.

Covers the backend-class selection boundary from issue #19615: valid
selection (including case-insensitive names), fallback diagnostics for
unknown/unavailable backends, preservation of the sparse/MLA/chunked
dispatch behavior, and the `create_attention` resolved-class pass-through.
No test here instantiates a real backend or touches CUDA.
"""

import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.attention.backends import utils as backend_utils
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.attention.backends.utils import (
    SUPPORTED_ATTENTION_BACKENDS,
    create_attention,
    get_attention_backend,
)
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttention

FLASHINFER_INSTALLED = importlib.util.find_spec("flashinfer") is not None


class _CapturingLogger:
    """Captures warning lines. tensorrt_llm's logger sets propagate=False, so
    the stdlib `caplog` fixture never sees them; swap the module's logger
    instead (see test_attention_failure_logging.py)."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.keys: list[str | None] = []

    def warning(self, *msg) -> None:
        self.messages.append(" ".join(str(m) for m in msg))
        self.keys.append(None)

    def warning_once(self, *msg, key=None) -> None:
        self.messages.append(" ".join(str(m) for m in msg))
        self.keys.append(key)


class _FakeBackend:
    """Stand-in backend class recording construction arguments."""

    support_mla_result = True

    def __init__(self, layer_idx, num_heads, head_dim, num_kv_heads=None, **kwargs) -> None:
        self.args = (layer_idx, num_heads, head_dim, num_kv_heads)
        self.kwargs = kwargs

    @classmethod
    def support_mla(cls) -> bool:
        return cls.support_mla_result


@pytest.fixture
def capturing_logger(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(backend_utils, "logger", logger)
    return logger


@pytest.mark.parametrize(
    "name, expected",
    [
        ("VANILLA", VanillaAttention),
        ("vanilla", VanillaAttention),
        ("Vanilla", VanillaAttention),
        ("TRTLLM", TrtllmAttention),
        ("trtllm", TrtllmAttention),
    ],
)
def test_dense_backends_resolve_without_fallback(name, expected, capturing_logger) -> None:
    assert get_attention_backend(name) is expected
    assert capturing_logger.messages == []


@pytest.mark.skipif(not FLASHINFER_INSTALLED, reason="requires the flashinfer package")
def test_flashinfer_resolves_when_available(monkeypatch, capturing_logger) -> None:
    from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttention

    monkeypatch.setattr(backend_utils, "IS_FLASHINFER_AVAILABLE", True)
    assert get_attention_backend("FLASHINFER") is FlashInferAttention
    assert capturing_logger.messages == []


def test_unknown_backend_falls_back_to_trtllm(capturing_logger) -> None:
    assert get_attention_backend("FLASHINFR") is TrtllmAttention
    assert len(capturing_logger.messages) == 1
    message = capturing_logger.messages[0]
    assert "'FLASHINFR'" in message
    assert "not a supported attention backend" in message
    assert "VANILLA, TRTLLM, FLASHINFER" in message
    assert "using TRTLLM instead" in message


@pytest.mark.parametrize("name", ["FLASHINFER", "flashinfer"])
def test_flashinfer_unavailable_falls_back_to_trtllm(name, monkeypatch, capturing_logger) -> None:
    monkeypatch.setattr(backend_utils, "IS_FLASHINFER_AVAILABLE", False)
    assert get_attention_backend(name) is TrtllmAttention
    assert len(capturing_logger.messages) == 1
    message = capturing_logger.messages[0]
    assert f"{name!r}" in message
    assert "FlashInfer package" in message
    assert "not installed" in message
    assert "using TRTLLM instead" in message


def test_fallback_reasons_use_distinct_stable_keys(monkeypatch, capturing_logger) -> None:
    get_attention_backend("BOGUS")
    get_attention_backend("OTHER")
    monkeypatch.setattr(backend_utils, "IS_FLASHINFER_AVAILABLE", False)
    get_attention_backend("FLASHINFER")
    unknown_keys = capturing_logger.keys[:2]
    unavailable_key = capturing_logger.keys[2]
    assert unknown_keys[0] == unknown_keys[1]
    assert unavailable_key != unknown_keys[0]


@pytest.mark.parametrize("name", ["VANILLA", "TRTLLM"])
def test_sparse_unknown_algorithm_still_raises(name, monkeypatch, capturing_logger) -> None:
    # The bogus algorithm reaches the sparse registry only when backend
    # selection itself succeeds; the unavailable-FLASHINFER fallback stays
    # dense without redispatch (covered above).
    monkeypatch.setattr(backend_utils, "IS_FLASHINFER_AVAILABLE", False)
    with pytest.raises(ValueError, match="bogus-algo"):
        get_attention_backend(name, sparse_params=SimpleNamespace(algorithm="bogus-algo"))


def test_flashinfer_unavailable_sparse_falls_back_dense_without_redispatch(
    monkeypatch, capturing_logger
) -> None:
    """FLASHINFER+unavailable with sparse params falls back to dense TRTLLM.

    Sparse configs are not redispatched through the TRTLLM sparse registry
    (see test_sparse_attention_backend_fallback_does_not_redispatch); the
    fallback only adds diagnostics naming request, reason, and selection.
    """
    monkeypatch.setattr(backend_utils, "IS_FLASHINFER_AVAILABLE", False)
    with patch(
        "tensorrt_llm._torch.attention.backends.utils.get_trtllm_sparse_attn_attention_backend"
    ) as trtllm_sparse_resolver:
        backend = get_attention_backend(
            "FLASHINFER", sparse_params=SimpleNamespace(algorithm="rocket")
        )
    assert backend is TrtllmAttention
    trtllm_sparse_resolver.assert_not_called()
    assert len(capturing_logger.messages) == 1
    message = capturing_logger.messages[0]
    assert "'FLASHINFER'" in message
    assert "FlashInfer package" in message
    assert "using TRTLLM instead" in message


def test_sparse_rocket_still_resolves() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.rocket import (
        RocketTrtllmAttention,
        RocketVanillaAttention,
    )

    assert get_attention_backend("TRTLLM", sparse_params=SimpleNamespace(algorithm="rocket")) is (
        RocketTrtllmAttention
    )
    assert get_attention_backend("VANILLA", sparse_params=SimpleNamespace(algorithm="rocket")) is (
        RocketVanillaAttention
    )


def test_create_attention_uses_provided_class_without_resolving(
    monkeypatch, capturing_logger
) -> None:
    def fail_if_resolving(*args, **kwargs):
        raise AssertionError("must not resolve when attn_cls is provided")

    monkeypatch.setattr(backend_utils, "get_attention_backend", fail_if_resolving)
    backend = create_attention("TRTLLM", 3, 8, 64, attn_cls=_FakeBackend)
    assert isinstance(backend, _FakeBackend)
    assert backend.args == (3, 8, 64, None)
    assert capturing_logger.messages == []


def test_create_attention_resolves_when_class_omitted(monkeypatch) -> None:
    calls = []

    def recording_resolve(backend_name, sparse_params=None):
        calls.append((backend_name, sparse_params))
        return _FakeBackend

    monkeypatch.setattr(backend_utils, "get_attention_backend", recording_resolve)
    backend = create_attention("TRTLLM", 0, 8, 64)
    assert isinstance(backend, _FakeBackend)
    assert calls == [("TRTLLM", None)]


def test_create_attention_chunked_check_runs_before_construction() -> None:
    with pytest.raises(ValueError, match="chunked attention"):
        create_attention("VANILLA", 0, 8, 64, attention_chunk_size=128, attn_cls=_FakeBackend)
    # The check is name-based, so a provided class does not bypass it, and a
    # chunked TRTLLM request still constructs.
    backend = create_attention("TRTLLM", 0, 8, 64, attention_chunk_size=128, attn_cls=_FakeBackend)
    assert isinstance(backend, _FakeBackend)


def test_create_attention_mla_check_uses_provided_class(monkeypatch) -> None:
    mla_kwargs = dict(
        q_lora_rank=4, kv_lora_rank=4, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4
    )
    monkeypatch.setattr(_FakeBackend, "support_mla_result", False)
    with pytest.raises(AssertionError, match="MLA is not supported"):
        create_attention(
            "TRTLLM", 0, 8, 64, is_mla_enable=True, attn_cls=_FakeBackend, **mla_kwargs
        )
    monkeypatch.setattr(_FakeBackend, "support_mla_result", True)
    backend = create_attention(
        "TRTLLM", 0, 8, 64, is_mla_enable=True, attn_cls=_FakeBackend, **mla_kwargs
    )
    assert isinstance(backend, _FakeBackend)


def test_supported_backend_vocabulary() -> None:
    assert tuple(SUPPORTED_ATTENTION_BACKENDS) == ("VANILLA", "TRTLLM", "FLASHINFER")
