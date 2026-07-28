# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from cache_dit import ForwardPattern
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_npu_available
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.cache.cache_dit_backend import CacheDiTAdapterConfig
from vllm_omni.quantization.component_config import safe_quant_config

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput, SequenceParallelOutput
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.adalayernorm import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk

logger = init_logger(__name__)


def _is_xpu_device() -> bool:
    """Check if current device is Intel XPU."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _join_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _get_tensor_parallel_size(parallel_config: DiffusionParallelConfig | None) -> int:
    if parallel_config is not None:
        return int(parallel_config.tensor_parallel_size)
    return get_tensor_model_parallel_world_size()


def _use_sharded_single_block_path(parallel_config: DiffusionParallelConfig | None) -> bool:
    # XPU-specific optimization: select sharded path for better performance on Intel XPU.
    # Only enable for XPU devices and when tensor parallelism is available.
    if not _is_xpu_device():
        return False
    return _get_tensor_parallel_size(parallel_config) > 1


@torch._dynamo.disable
def _apply_row_parallel_local(layer: RowParallelLinear, x: torch.Tensor) -> torch.Tensor:
    bias = layer.bias if get_tensor_model_parallel_rank() == 0 else None
    return layer.quant_method.apply(layer, x, bias)


class ColumnParallelApproxGELU(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        approximate: str,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in,
            dim_out,
            bias=bias,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "proj"),
        )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


class FluxSingleBlockOutput(nn.Module):
    def __init__(
        self,
        attn_dim: int,
        mlp_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attn_dim = attn_dim
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.attn_proj = RowParallelLinear(
            attn_dim,
            out_dim,
            bias=bias,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "attn_proj"),
        )
        self.mlp_proj = RowParallelLinear(
            mlp_dim,
            out_dim,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "mlp_proj"),
        )

    @torch._dynamo.disable
    def forward(self, attn_hidden_states: torch.Tensor, mlp_hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = _apply_row_parallel_local(self.attn_proj, attn_hidden_states)
        mlp_output = _apply_row_parallel_local(self.mlp_proj, mlp_hidden_states)
        hidden_states = attn_output + mlp_output
        if get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return hidden_states

    def load_weight(self, weight_name: str, loaded_weight: torch.Tensor) -> None:
        if weight_name == "bias":
            param = self.attn_proj.bias
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            return

        self._load_split_weight(self.attn_proj, weight_name, loaded_weight, self.attn_dim)
        self._load_split_weight(self.mlp_proj, weight_name, loaded_weight, self.mlp_dim, start_dim=self.attn_dim)

    def _load_split_weight(
        self,
        proj: RowParallelLinear,
        weight_name: str,
        loaded_weight: torch.Tensor,
        logical_width: int,
        *,
        start_dim: int = 0,
    ) -> None:
        param = getattr(proj, weight_name)
        split_dim = getattr(param, "input_dim", 0)
        total_width = self.attn_dim + self.mlp_dim
        dim_size = loaded_weight.shape[split_dim]
        start_idx = dim_size * start_dim // total_width
        shard_size = dim_size * logical_width // total_width
        loaded_weight = loaded_weight.narrow(split_dim, start_idx, shard_size)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)

    def loaded_parameter_names(self, weight_name: str) -> list[str]:
        if weight_name == "bias":
            return ["attn_proj.bias"]
        return [f"attn_proj.{weight_name}", f"mlp_proj.{weight_name}"]


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        activation_fn: str = "gelu-approximate",
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert activation_fn == "gelu-approximate", "Only gelu-approximate is supported."

        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(
                dim,
                inner_dim,
                approximate="tanh",
                bias=bias,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "net.0"),
            ),
            nn.Identity(),  # placeholder for weight loading
            RowParallelLinear(
                inner_dim,
                dim_out,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "net.2"),
            ),
        ]

        self.net = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class FluxAttention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        output_is_parallel: bool = False,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.output_is_parallel = output_is_parallel
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "to_qkv"),
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList(
                [
                    RowParallelLinear(
                        self.inner_dim,
                        self.out_dim,
                        bias=out_bias,
                        input_is_parallel=True,
                        return_bias=False,
                        quant_config=quant_config,
                        prefix=_join_prefix(prefix, "to_out.0"),
                    ),
                    nn.Dropout(dropout),
                ]
            )

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=self.added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                bias=added_proj_bias,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "add_kv_proj"),
            )
            self.to_add_out = RowParallelLinear(
                self.inner_dim,
                query_dim,
                bias=out_bias,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "to_add_out"),
            )

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        precomputed_qkv: torch.Tensor | None = kwargs.get("precomputed_qkv")
        if precomputed_qkv is None:
            qkv, _ = self.to_qkv(hidden_states)
        else:
            qkv = precomputed_qkv
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
            add_q_size = self.add_kv_proj.num_heads * self.head_dim
            add_kv_size = self.add_kv_proj.num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )

            encoder_query = encoder_query.unflatten(-1, (self.add_kv_proj.num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # SP-aware single-stream path: attend from image tokens with text as joint context.
        # XPU-specific optimization enabled only for Intel XPU devices
        text_seq_len = kwargs.get("text_seq_len")
        use_sp_single_stream = bool(kwargs.get("use_sp_single_stream", False))
        is_xpu = hidden_states.device.type == "xpu"
        if (
            encoder_hidden_states is None
            and use_sp_single_stream
            and is_xpu
            and text_seq_len is not None
            and image_rotary_emb is not None
        ):
            cos, sin = image_rotary_emb
            if cos.dtype != query.dtype:
                cos = cos.to(query.dtype)
            if sin.dtype != query.dtype:
                sin = sin.to(query.dtype)
            txt_cos, img_cos = cos[:text_seq_len], cos[text_seq_len:]
            txt_sin, img_sin = sin[:text_seq_len], sin[text_seq_len:]

            img_query = query[:, text_seq_len:]
            img_key = key[:, text_seq_len:]
            img_value = value[:, text_seq_len:]
            text_query = query[:, :text_seq_len]
            text_key = key[:, :text_seq_len]
            text_value = value[:, :text_seq_len]

            img_query = self.rope(img_query, img_cos, img_sin)
            img_key = self.rope(img_key, img_cos, img_sin)
            text_query = self.rope(text_query, txt_cos, txt_sin)
            text_key = self.rope(text_key, txt_cos, txt_sin)

            attn_metadata = AttentionMetadata(
                joint_query=text_query,
                joint_key=text_key,
                joint_value=text_value,
                joint_strategy="front",
            )
            hidden_states_mask: torch.Tensor | None = kwargs.get("hidden_states_mask", None)
            encoder_hidden_states_mask: torch.Tensor | None = kwargs.get("encoder_hidden_states_mask", None)
            if hidden_states_mask is not None:
                attn_metadata.attn_mask = hidden_states_mask
            if encoder_hidden_states_mask is not None:
                attn_metadata.joint_attn_mask = encoder_hidden_states_mask

            hidden_states = self.attn(img_query, img_key, img_value, attn_metadata)
        else:
            query, key = apply_rope_to_qk(self.rope, query, key, image_rotary_emb)  # [S, D/2]

            attn_metadata = None
            if attention_mask is not None:
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                attn_metadata = AttentionMetadata(attn_mask=attention_mask)

            hidden_states = self.attn(
                query,
                key,
                value,
                attn_metadata,
            )
        hidden_states = hidden_states.flatten(2, 3)
        if hidden_states.dtype != query.dtype:
            hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states.split(
                [encoder_seq_len, hidden_states.shape[1] - encoder_seq_len],
                dim=1,
            )
            hidden_states = self.to_out[0](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
            hidden_states = self.to_out[1](hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            if get_tensor_model_parallel_world_size() > 1 and not self.output_is_parallel:
                hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states


class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim, quant_config=quant_config, prefix=_join_prefix(prefix, "norm1"))
        self.norm1_context = AdaLayerNormZero(
            dim,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "norm1_context"),
        )

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "attn"),
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config, prefix=_join_prefix(prefix, "ff"))

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(
            dim=dim,
            dim_out=dim,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "ff_context"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
        parallel_config: DiffusionParallelConfig | None = None,
    ):
        super().__init__()
        self.parallel_config = parallel_config
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.use_sharded_single_block = _use_sharded_single_block_path(parallel_config)

        self.norm = AdaLayerNormZeroSingle(
            dim,
            quant_config=safe_quant_config(quant_config),
            prefix=_join_prefix(prefix, "norm"),
        )
        if self.use_sharded_single_block:
            self.proj_mlp = ColumnParallelApproxGELU(
                dim,
                self.mlp_hidden_dim,
                approximate="tanh",
                bias=True,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "proj_mlp"),
            )
            self.proj_out = FluxSingleBlockOutput(
                dim,
                self.mlp_hidden_dim,
                dim,
                bias=True,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "proj_out"),
            )
        else:
            self.proj_mlp = ReplicatedLinear(
                dim,
                self.mlp_hidden_dim,
                bias=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "proj_mlp"),
            )
            self.act_mlp = nn.GELU(approximate="tanh")
            self.proj_out = ReplicatedLinear(
                dim + self.mlp_hidden_dim,
                dim,
                bias=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "proj_out"),
            )

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            output_is_parallel=self.use_sharded_single_block,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "attn"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # XPU-specific optimization: sequence parallel single stream
        is_xpu = hidden_states.device.type == "xpu"
        sp_size = self.parallel_config.sequence_parallel_size if self.parallel_config is not None else None
        use_sp_single_stream = (
            is_xpu
            and sp_size is not None
            and sp_size > 1
            and is_forward_context_available()
            and not get_forward_context().split_text_embed_in_sp
        )

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        qkv, mlp_hidden_states = self.to_qkv_mlp_proj(norm_hidden_states)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            text_seq_len=text_seq_len,
            use_sp_single_stream=use_sp_single_stream,
            precomputed_qkv=qkv,
            **joint_attention_kwargs,
        )
        if self.use_sharded_single_block:
            hidden_states = gate.unsqueeze(1) * self.proj_out(attn_output, mlp_hidden_states)
        else:
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states

    def to_qkv_mlp_proj(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Project single-stream inputs for QKV and MLP in one path.
        qkv, _ = self.attn.to_qkv(hidden_states)
        if self.use_sharded_single_block:
            mlp_hidden_states = self.proj_mlp(hidden_states)
        else:
            mlp_hidden_states = self.act_mlp(self.proj_mlp(hidden_states))
        return qkv, mlp_hidden_states


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor, target_dtype: torch.dtype | None = None) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        is_xpu = ids.device.type == "xpu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            freqs_cis = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                use_real=False,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1)
        freqs_sin = torch.cat(sin_out, dim=-1)

        # XPU-specific optimization: fused device + dtype conversion to avoid intermediate copy
        if is_xpu and target_dtype is not None:
            if freqs_cos.device != ids.device or freqs_cos.dtype != target_dtype:
                freqs_cos = freqs_cos.to(device=ids.device, dtype=target_dtype)
                freqs_sin = freqs_sin.to(device=ids.device, dtype=target_dtype)
        elif target_dtype is None:
            if freqs_cos.device != ids.device:
                freqs_cos = freqs_cos.to(ids.device)
                freqs_sin = freqs_sin.to(ids.device)
        else:
            # Standard path for non-XPU devices
            if freqs_cos.device != ids.device or freqs_cos.dtype != target_dtype:
                freqs_cos = freqs_cos.to(device=ids.device, dtype=target_dtype)
                freqs_sin = freqs_sin.to(device=ids.device, dtype=target_dtype)
        return freqs_cos, freqs_sin


class FluxTransformer2DModel(nn.Module):
    """
    The Transformer model introduced in Flux.

    Args:
        od_config (`OmniDiffusionConfig`):
            The configuration for the model.
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={
            "transformer_blocks": ForwardPattern.Pattern_1,
            "single_transformer_blocks": ForwardPattern.Pattern_1,
        }
    )

    # the small and frequently-repeated block(s) of a model
    # -- typically a transformer layer
    # used for torch compile optimizations
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]
    _sp_plan = {
        "": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return ("transformer_blocks" in name or "single_transformer_blocks" in name) and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        theta: float = 10000.0,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        if od_config is not None:
            model_config = od_config.tf_model_config
            num_layers = model_config.num_layers
            self.parallel_config = od_config.parallel_config
        else:
            self.parallel_config = None

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.guidance_embeds = guidance_embeds

        self.pos_embed = FluxPosEmbed(theta=theta, axes_dim=axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=safe_quant_config(quant_config),
                    prefix=f"transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    parallel_config=self.parallel_config,
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=quant_config,
                    prefix=f"single_transformer_blocks.{i}",
                )
                for i in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            quant_config=safe_quant_config(quant_config),
            prefix="norm_out",
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            img_ids: (`torch.Tensor`):
                The position ids for image tokens.
            txt_ids (`torch.Tensor`):
                The position ids for text tokens.
            guidance (`torch.Tensor`):
                Guidance embeddings for guidance-distilled variant of the model.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        if self.parallel_config is not None:
            sp_size = self.parallel_config.sequence_parallel_size
            if sp_size is not None and sp_size > 1 and is_forward_context_available():
                get_forward_context().split_text_embed_in_sp = False

        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.pos_embed(ids, target_dtype=hidden_states.dtype)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            # cross-attn
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        # we need to load the buffers for beta and eps (XIELU)
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if (
                    lookup_name not in params_dict
                    and "single_transformer_blocks." in lookup_name
                    and ".proj_mlp." in lookup_name
                ):
                    lookup_name = lookup_name.replace(".proj_mlp.", ".proj_mlp.proj.")
                elif "single_transformer_blocks." in lookup_name and ".proj_out." in lookup_name:
                    module_name, _, weight_name = lookup_name.rpartition(".proj_out.")
                    proj_out_module = self.get_submodule(f"{module_name}.proj_out")
                    if hasattr(proj_out_module, "load_weight") and hasattr(proj_out_module, "loaded_parameter_names"):
                        proj_out_module.load_weight(weight_name, loaded_weight)
                        loaded_params.add(original_name)
                        loaded_params.add(lookup_name)
                        for suffix in proj_out_module.loaded_parameter_names(weight_name):
                            loaded_params.add(f"{module_name}.proj_out.{suffix}")
                        continue

                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params


class FluxKontextTransformer2DModel(FluxTransformer2DModel):
    def __init__(
        self,
        od_config: OmniDiffusionConfig = None,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        theta: float = 10000.0,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__(
            od_config=od_config,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
            theta=theta,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

        if guidance is None:
            temb = self.time_text_embed(timestep, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, guidance, pooled_projections)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids is not None and txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch.Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids is not None and img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch.Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids, target_dtype=hidden_states.dtype)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
