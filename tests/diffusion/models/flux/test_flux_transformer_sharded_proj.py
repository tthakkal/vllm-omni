# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.flux.flux_transformer import (
    ColumnParallelApproxGELU,
    FluxSingleBlockOutput,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    _should_use_flux_optimizations,
    _use_sharded_single_block_path,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _init_distributed():
    """Initialize minimal distributed state for TP-aware linear layers.

    ``world_size=1``: these tests cover branch selection and weight-loading
    bookkeeping only. ``tensor_parallel_size=2`` below selects the sharded
    branch but no collective, per-rank shard or all-reduce actually runs.
    Real two-rank forward/load parity lives in
    ``tests/diffusion/distributed/test_flux_sharded_proj_tp2.py``.
    """
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29513",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "env_value,expected",
    [
        (None, False),
        ("", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("off", False),
        ("disabled", False),
        ("unexpected", False),
    ],
)
def test_should_use_flux_optimizations_env_values(monkeypatch, env_value, expected):
    env_key = "VLLM_OMNI_FLUX1_SHARDED_PROJ"
    if env_value is None:
        monkeypatch.delenv(env_key, raising=False)
    else:
        monkeypatch.setenv(env_key, env_value)

    assert _should_use_flux_optimizations() is expected


def test_use_sharded_single_block_path_respects_env_and_tp(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")

    tp2 = DiffusionParallelConfig(tensor_parallel_size=2)
    tp1 = DiffusionParallelConfig(tensor_parallel_size=1)

    assert _use_sharded_single_block_path(tp2) is True
    assert _use_sharded_single_block_path(tp1) is False

    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "0")
    assert _use_sharded_single_block_path(tp2) is False


def test_use_sharded_single_block_path_uses_runtime_tp_when_config_missing(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.flux.flux_transformer.get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    assert _use_sharded_single_block_path(None) is True

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.flux.flux_transformer.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    assert _use_sharded_single_block_path(None) is False


def test_single_transformer_block_uses_sharded_modules_when_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")
    block = FluxSingleTransformerBlock(
        dim=64,
        num_attention_heads=2,
        attention_head_dim=32,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
        prefix="single_transformer_blocks.0",
    )

    assert block.use_sharded_single_block is True
    assert isinstance(block.proj_mlp, ColumnParallelApproxGELU)
    assert isinstance(block.proj_out, FluxSingleBlockOutput)
    assert block.attn.output_is_parallel is True


def test_single_transformer_block_uses_replicated_modules_when_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "0")
    block = FluxSingleTransformerBlock(
        dim=64,
        num_attention_heads=2,
        attention_head_dim=32,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
        prefix="single_transformer_blocks.0",
    )

    assert block.use_sharded_single_block is False
    assert isinstance(block.proj_out, ReplicatedLinear)
    assert hasattr(block, "act_mlp")
    assert block.attn.output_is_parallel is False


def test_load_split_weight_replicates_non_sharded_scale_tensor():
    module = object.__new__(FluxSingleBlockOutput)
    module.attn_dim = 4
    module.mlp_dim = 8

    calls = []

    class Param:
        pass

    param = Param()

    def _loader(p, w):
        calls.append((p, w.clone()))

    param.weight_loader = _loader
    proj = SimpleNamespace(scale=param)
    loaded_weight = torch.arange(6, dtype=torch.float32)

    module._load_split_weight(proj, "scale", loaded_weight, logical_width=4)

    assert len(calls) == 1
    assert calls[0][0] is param
    torch.testing.assert_close(calls[0][1], loaded_weight)


def test_load_split_weight_narrows_by_logical_width_and_start_dim():
    module = object.__new__(FluxSingleBlockOutput)
    module.attn_dim = 4
    module.mlp_dim = 8

    calls = []

    class Param:
        input_dim = 0

    param = Param()

    def _loader(p, w):
        calls.append((p, w.clone()))

    param.weight_loader = _loader
    proj = SimpleNamespace(weight=param)
    loaded_weight = torch.arange(24, dtype=torch.float32).reshape(12, 2)

    module._load_split_weight(proj, "weight", loaded_weight, logical_width=4, start_dim=0)
    module._load_split_weight(proj, "weight", loaded_weight, logical_width=8, start_dim=4)

    assert len(calls) == 2
    torch.testing.assert_close(calls[0][1], loaded_weight[0:4])
    torch.testing.assert_close(calls[1][1], loaded_weight[4:12])


def test_load_split_weight_raises_for_missing_parameter():
    module = object.__new__(FluxSingleBlockOutput)
    module.attn_dim = 4
    module.mlp_dim = 8
    proj = SimpleNamespace()

    with pytest.raises(ValueError, match="VLLM_OMNI_FLUX1_SHARDED_PROJ"):
        module._load_split_weight(proj, "weight", torch.ones(4, 4), logical_width=4)


def test_tp2_load_weights_splits_proj_out_weight_in_sharded_path(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")

    od_config = SimpleNamespace(
        tf_model_config=SimpleNamespace(num_layers=0),
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
    )
    model = FluxTransformer2DModel(
        od_config=od_config,
        in_channels=4,
        num_layers=0,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        axes_dims_rope=(4, 6, 6),
        guidance_embeds=False,
    )

    block = model.single_transformer_blocks[0]
    assert block.use_sharded_single_block is True
    assert isinstance(block.proj_out, FluxSingleBlockOutput)

    proj_out = block.proj_out
    attn_dim = proj_out.attn_dim
    mlp_dim = proj_out.mlp_dim
    total_dim = attn_dim + mlp_dim

    loaded_weight = torch.randn(proj_out.out_dim, total_dim)
    loaded = model.load_weights(
        [
            ("single_transformer_blocks.0.proj_out.weight", loaded_weight),
        ]
    )

    assert "single_transformer_blocks.0.proj_out.weight" in loaded
    assert "single_transformer_blocks.0.proj_out.attn_proj.weight" in loaded
    assert "single_transformer_blocks.0.proj_out.mlp_proj.weight" in loaded

    attn_param = proj_out.attn_proj.weight
    mlp_param = proj_out.mlp_proj.weight
    split_dim = attn_param.input_dim
    dim_size = loaded_weight.shape[split_dim]

    attn_start = 0
    attn_size = dim_size * attn_dim // total_dim
    mlp_start = dim_size * attn_dim // total_dim
    mlp_size = dim_size * mlp_dim // total_dim

    expected_attn = loaded_weight.narrow(split_dim, attn_start, attn_size)
    expected_mlp = loaded_weight.narrow(split_dim, mlp_start, mlp_size)

    torch.testing.assert_close(attn_param.detach(), expected_attn)
    torch.testing.assert_close(mlp_param.detach(), expected_mlp)


def test_tp2_quantized_checkpoint_scale_is_replicated_for_sharded_proj_out(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")

    block = FluxSingleTransformerBlock(
        dim=64,
        num_attention_heads=2,
        attention_head_dim=32,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
        prefix="single_transformer_blocks.0",
    )

    assert block.use_sharded_single_block is True
    proj_out = block.proj_out
    assert isinstance(proj_out, FluxSingleBlockOutput)

    calls = []

    class FakeScaleParam:
        pass

    def _loader(param, loaded_weight):
        calls.append((param, loaded_weight.clone()))

    attn_scale = FakeScaleParam()
    mlp_scale = FakeScaleParam()
    attn_scale.weight_loader = _loader
    mlp_scale.weight_loader = _loader

    # Quantized scale tensors carry no input_dim and should be replicated.
    setattr(proj_out.attn_proj, "weight_scale", attn_scale)
    setattr(proj_out.mlp_proj, "weight_scale", mlp_scale)

    scale_weight = torch.randn(17)
    proj_out.load_weight("weight_scale", scale_weight)

    assert len(calls) == 2
    torch.testing.assert_close(calls[0][1], scale_weight)
    torch.testing.assert_close(calls[1][1], scale_weight)
