# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Platform gating of the attention-mask capability flags.

``supports_piecewise_spans`` and ``supports_dense_attention_mask`` describe what the
``forward_*`` that actually runs can do, not what the backend class can do. Declaring
support on a platform whose forward ignores the feature makes models skip building the
equivalent ``attn_mask`` and run unmasked dense attention with no error and no warning,
so these flags are asserted per platform rather than once per class.
"""

import sys
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.attention.backends.flash_attn_hub import (
    FlashAttention3HubBackend,
    FlashAttentionHubBackend,
)
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Modules that resolve ``current_omni_platform`` at call time and must be patched
# independently: the plain flash backend and the two hub backends live in different
# modules, and a fix applied to only one of them would leave the others lying.
_BACKEND_MODULES = {
    FlashAttentionBackend: "vllm_omni.diffusion.attention.backends.flash_attn",
    FlashAttentionHubBackend: "vllm_omni.diffusion.attention.backends.flash_attn_hub",
    FlashAttention3HubBackend: "vllm_omni.diffusion.attention.backends.flash_attn_hub",
}

_PLATFORMS = ("cuda", "rocm", "musa", "xpu", "npu")


class _Platform(OmniPlatform):
    """A real ``OmniPlatform`` selected by enum, not a duck-typed stand-in.

    Every ``is_*`` predicate the flags call is inherited from ``OmniPlatform`` and derived
    from ``_omni_enum``, so this exercises the same accessors a live platform uses and
    cannot drift from them.
    """

    def __init__(self, name: str) -> None:
        self._omni_enum = OmniPlatformEnum(name)


@dataclass
class _Layer:
    """The one ``Attention`` field these unbound helpers read.

    Typed on purpose: a real ``Attention`` needs a full backend selection to construct,
    and ``_mask_needs_sdpa`` / ``_assert_piecewise_compatible`` touch nothing else.
    """

    attn_backend: type[AttentionBackend]


@pytest.fixture
def as_platform(monkeypatch):
    """Evaluate a backend's flags as if running on ``name``."""

    def _run(backend: type[AttentionBackend], name: str, flag: str) -> bool:
        assert name in _PLATFORMS, name
        monkeypatch.setattr(f"{_BACKEND_MODULES[backend]}.current_omni_platform", _Platform(name))
        return getattr(backend, flag)()

    return _run


@pytest.mark.parametrize("backend", list(_BACKEND_MODULES))
@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        # Only forward_cuda dispatches piecewise_attn. ROCm and MUSA reach it through
        # the default forward_hip / forward_musa delegation.
        ("cuda", True),
        ("rocm", True),
        ("musa", True),
        # forward_xpu and the NPU forwards never read full_attn_spans.
        ("xpu", False),
        ("npu", False),
    ],
)
def test_piecewise_spans_gated_to_cuda_family(as_platform, backend, platform, expected):
    assert as_platform(backend, platform, "supports_piecewise_spans") is expected


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_dense_attention_mask_false_only_on_xpu(as_platform, platform):
    """XPU is the one platform that has to reroute; everything else is unchanged.

    ``forward_xpu``'s only mask route is ``_forward_varlen_masked``, which asserts a 2D
    ``[batch, seq]`` padding mask, and it has no piecewise path to carry the pattern
    instead. The CUDA family keeps True deliberately even though it honors a >2D mask
    only via ``piecewise_attn``: declaring False there would drag HunyuanImage3 off
    ``piecewise_attn`` onto SDPA. NPU passes 4D masks straight to the Ascend kernel.
    """
    expected = platform != "xpu"
    assert as_platform(FlashAttentionBackend, platform, "supports_dense_attention_mask") is expected


@pytest.mark.parametrize("backend", [FlashAttentionHubBackend, FlashAttention3HubBackend])
@pytest.mark.parametrize("platform", _PLATFORMS)
def test_hub_backends_keep_dense_mask_support_everywhere(as_platform, backend, platform):
    """The hub backends stay True on every platform, including XPU, on purpose.

    They implement only ``forward_cuda``, so on XPU the base-class ``forward_xpu``
    raises ``NotImplementedError`` and the flag is never consulted for a real dispatch.
    Declaring False to be tidy would cost real behavior on CUDA, where it would divert
    HunyuanImage3 -- which passes a 4D mask together with ``full_attn_spans`` -- off
    ``piecewise_attn`` and onto SDPA.
    """
    assert as_platform(backend, platform, "supports_dense_attention_mask") is True


@pytest.mark.parametrize("backend", list(_BACKEND_MODULES))
def test_rocm_and_musa_reach_forward_cuda(backend):
    """The rocm/musa span claim rests on this delegation, so pin it.

    ``supports_piecewise_spans`` answers True for ROCm and MUSA even though neither
    backend defines ``forward_hip`` / ``forward_musa``; that is only sound while the
    base class delegates both to ``forward_cuda``.
    """
    impl = backend.get_impl_cls()
    assert impl.forward_hip is AttentionImpl.forward_hip
    assert impl.forward_musa is AttentionImpl.forward_musa
    assert impl.forward_cuda is not AttentionImpl.forward_cuda


def test_piecewise_spans_is_a_call_not_a_truthy_attribute():
    """Guards against the regression class this fix exists for.

    A bare ``if not backend.supports_piecewise_spans:`` on a bound method is always
    False, so the check silently never fires. Asserting the attribute is callable and
    that the base class default is a real ``False`` keeps a future refactor from
    reverting it to a class attribute.
    """
    assert callable(AttentionBackend.supports_piecewise_spans)
    assert AttentionBackend.supports_piecewise_spans() is False
    assert callable(AttentionBackend.supports_dense_attention_mask)


class _MaskAware(AttentionBackend):
    """Reports mask support but only accepts a 2D padding mask."""

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @classmethod
    def supports_dense_attention_mask(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "MASK_AWARE_2D_ONLY"


class _MaskDense(_MaskAware):
    @classmethod
    def supports_dense_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "MASK_AWARE_DENSE"


class _MaskBlind(AttentionBackend):
    """Reports no mask support, like TRTLLM_ATTN / SAGE_ATTN / RAINFUSION_ATTN."""

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "MASK_BLIND"


def test_dense_mask_default_tracks_supports_attention_mask():
    """A backend that only implements ``supports_attention_mask`` must not opt in silently."""
    assert _MaskBlind.supports_dense_attention_mask() is False


@pytest.mark.parametrize(
    ("backend", "mask", "expected"),
    [
        # The case this fix is for: a 4D mask on a backend that can only take 2D.
        (_MaskAware, torch.ones(1, 1, 4, 4, dtype=torch.bool), True),
        (_MaskAware, torch.ones(1, 2, 4, 4, dtype=torch.bool), True),
        # 3D is still more than the 2D assert accepts.
        (_MaskAware, torch.ones(1, 4, 4, dtype=torch.bool), True),
        # A 2D padding mask is exactly what the varlen route wants; no reroute.
        (_MaskAware, torch.ones(1, 4, dtype=torch.bool), False),
        (_MaskAware, None, False),
        # A backend that honors the dense mask keeps it.
        (_MaskDense, torch.ones(1, 1, 4, 4, dtype=torch.bool), False),
        # Scoped out: rerouting here would preempt TRTLLM_ATTN's deliberate ValueError.
        (_MaskBlind, torch.ones(1, 1, 4, 4, dtype=torch.bool), False),
    ],
)
def test_mask_needs_sdpa_truth_table(backend, mask, expected):
    layer = _Layer(backend)
    metadata = AttentionMetadata(attn_mask=mask)
    assert Attention._mask_needs_sdpa(layer, metadata) is expected


def test_mask_needs_sdpa_without_metadata():
    layer = _Layer(_MaskAware)
    assert Attention._mask_needs_sdpa(layer, None) is False


def test_piecewise_assert_ignores_absent_spans():
    layer = _Layer(_MaskBlind)
    Attention._assert_piecewise_compatible(layer, AttentionMetadata(full_attn_spans=None))
    Attention._assert_piecewise_compatible(layer, None)


@pytest.mark.parametrize("spans", [[[(0, 4)]], [], [[]], [[], []]])
def test_piecewise_assert_rejects_any_non_none_spans_on_unsupported_backend(spans):
    """Only ``None`` bypasses this guard -- an empty span list still raises.

    Emptiness is not the unrestricted case: ``build_segments`` emits a single causal
    segment covering the whole sequence when it finds no spans, so ``[[]]`` asks for
    *more* masking than ``[[(0, n)]]``, not less. ``forward_cuda`` keys on ``is None``
    for the same reason.
    """
    layer = _Layer(_MaskBlind)
    metadata = AttentionMetadata(full_attn_spans=spans)
    with pytest.raises(ValueError, match="does not support piecewise attention"):
        Attention._assert_piecewise_compatible(layer, metadata)


def test_piecewise_assert_accepts_spans_with_4d_mask():
    """A 4D mask carries the pattern itself, so spans are advisory and allowed."""
    layer = _Layer(_MaskBlind)
    metadata = AttentionMetadata(
        full_attn_spans=[[(0, 4)]],
        attn_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
    )
    Attention._assert_piecewise_compatible(layer, metadata)


class _RecordingImpl:
    """Records which impl ``_run_local_attention`` dispatched to."""

    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def forward(self, query, key, value, attn_metadata):
        self.calls.append(self.name)
        return query


def _flash_layer(calls: list[str]) -> Attention:
    """A real ``Attention`` carrying only what ``_run_local_attention`` reads on this path.

    Built without ``__init__`` because that resolves a backend for the host platform; the point
    here is to run the real dispatch with FLASH_ATTN as resolved on the patched platform.
    """
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer.attn_backend = FlashAttentionBackend
    layer.attention = _RecordingImpl("flash", calls)
    layer.sdpa_fallback = _RecordingImpl("sdpa", calls)
    layer.allow_fp32_fallback = False
    layer.backend_explicit = False
    layer._scheduler_paged_kv = False
    layer.paged_kv_cache_role = None
    return layer


@pytest.mark.parametrize(
    ("platform", "mask", "spans", "expected"),
    [
        # The fix: XPU flash would hit forward_xpu's 2D assert, so >2D masks run on SDPA.
        # HunyuanImage3 / glm_image / flux / flux2 / ernie_image all pass 4D masks here.
        ("xpu", torch.ones(1, 1, 4, 4, dtype=torch.bool), None, "sdpa"),
        ("xpu", torch.ones(1, 1, 4, 4, dtype=torch.bool), [[(0, 4)]], "sdpa"),
        ("xpu", torch.ones(1, 4, 4, dtype=torch.bool), None, "sdpa"),
        # A 2D padding mask is what forward_xpu's varlen route accepts; keep flash.
        ("xpu", torch.ones(1, 4, dtype=torch.bool), None, "flash"),
        ("xpu", None, None, "flash"),
        # CUDA keeps HunyuanImage3's 4D mask + spans on flash so it reaches piecewise_attn.
        ("cuda", torch.ones(1, 1, 4, 4, dtype=torch.bool), [[(0, 4)]], "flash"),
        # NPU hands 4D masks to aclnnFlashAttentionScore directly.
        ("npu", torch.ones(1, 1, 4, 4, dtype=torch.bool), None, "flash"),
    ],
)
def test_run_local_attention_dispatches_dense_mask(monkeypatch, platform, mask, spans, expected):
    monkeypatch.setattr(f"{_BACKEND_MODULES[FlashAttentionBackend]}.current_omni_platform", _Platform(platform))
    calls: list[str] = []
    layer = _flash_layer(calls)
    q = torch.zeros(1, 4, 2, 8)
    metadata = AttentionMetadata(attn_mask=mask, full_attn_spans=spans)

    Attention._run_local_attention(layer, q, q, q, metadata)

    assert calls == [expected]


@pytest.mark.parametrize("backend_enum", list(DiffusionAttentionBackendEnum), ids=lambda e: e.name)
@pytest.mark.parametrize("platform", _PLATFORMS)
def test_every_registered_backend_answers_flags_as_callables(monkeypatch, backend_enum, platform):
    """No backend may shadow a capability classmethod with a plain class attribute.

    ``supports_piecewise_spans: bool = True`` on a subclass replaces the base classmethod with
    a ``bool``, so every ``supports_piecewise_spans()`` call raises ``TypeError`` -- and a
    caller that forgets the ``()`` instead reads a bound method, which is always truthy.
    """
    try:
        backend = backend_enum.get_class()
    except ImportError as exc:
        pytest.skip(f"{backend_enum.name} is not importable here: {exc}")
    module = backend.__module__
    if hasattr(sys.modules[module], "current_omni_platform"):
        monkeypatch.setattr(f"{module}.current_omni_platform", _Platform(platform))
    for flag in ("supports_piecewise_spans", "supports_dense_attention_mask"):
        assert isinstance(getattr(backend, flag)(), bool), f"{backend_enum.name}.{flag}"
