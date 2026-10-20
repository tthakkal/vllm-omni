# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU lock: the ``auto`` attention path calls the backend's span probe.

``supports_piecewise_spans`` is a classmethod. Reading it without ``()`` yields a bound
method, which is always truthy, so every backend -- including TORCH_SDPA and FLASH_ATTN on
XPU / NPU, whose forwards ignore ``full_attn_spans`` -- would be sent down the piecewise path.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21Transformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _SpansOn(AttentionBackend):
    @classmethod
    def supports_piecewise_spans(cls) -> bool:
        return True


class _SpansOff(AttentionBackend):
    @classmethod
    def supports_piecewise_spans(cls) -> bool:
        return False


def _model_with_backend(backend: type[AttentionBackend]) -> SimpleNamespace:
    """Just the ``transformer_blocks[0].attn.attn.attn_backend`` chain the resolver walks."""
    block = SimpleNamespace(attn=SimpleNamespace(attn=SimpleNamespace(attn_backend=backend)))
    return SimpleNamespace(causal_block=True, transformer_blocks=[block])


@pytest.mark.parametrize(("backend", "expected"), [(_SpansOn, "piecewise"), (_SpansOff, "mask")])
def test_auto_path_follows_backend_span_support(backend, expected):
    model = _model_with_backend(backend)
    path = QwenImage21Transformer2DModel._resolve_attn_path(model, "auto", has_padding=False, sp_padding=False)
    assert path == expected
