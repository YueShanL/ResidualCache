"""Differentiable historical block gating for Gemma 4 eager attention.

The inference replay adapter physically packs selected K/V.  Training cannot
differentiate through that discrete operation, so this adapter prepends every
candidate historical K/V and adds one learned log gate per complete block to
the historical attention logits.  Native local/sliding K/V are untouched.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from residual_cache.gemma4_memory_adapter import (
    Gemma4StaticKVController,
    eager_attention_reference,
)


ATTENTION_BACKEND_NAME = "output_preserving_region_gate"
_CONTROLLERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_CONTROLLER_LOCK = threading.RLock()
_BACKEND_REGISTERED = False


def _repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    batch, key_value_heads, tokens, head_dim = hidden_states.shape
    if repetitions == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, key_value_heads, repetitions, tokens, head_dim
    )
    return expanded.reshape(batch, key_value_heads * repetitions, tokens, head_dim)


def _prepend_native_mask(
    attention_mask: torch.Tensor | None,
    *,
    query: torch.Tensor,
    historical_length: int,
    native_key_length: int,
) -> torch.Tensor:
    if historical_length <= 0:
        raise ValueError("historical_length must be positive")
    if attention_mask is None:
        native = torch.zeros(
            (query.shape[0], 1, query.shape[2], native_key_length),
            device=query.device,
            dtype=query.dtype,
        )
    else:
        if attention_mask.ndim != 4:
            raise ValueError("Gemma eager attention requires a four-dimensional mask")
        if attention_mask.shape[-1] != native_key_length:
            raise ValueError("native attention mask does not match native K/V length")
        native = attention_mask
    prefix = torch.zeros(
        (native.shape[0], native.shape[1], native.shape[2], historical_length),
        device=native.device,
        dtype=native.dtype,
    )
    return torch.cat((prefix, native), dim=-1)


class Gemma4SoftBlockGateController:
    def __init__(
        self,
        layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
        block_gates: torch.Tensor,
        block_token_counts: Sequence[int],
        *,
        gate_epsilon: float,
    ) -> None:
        self.layer_kv = {int(layer): pair for layer, pair in layer_kv.items()}
        if not self.layer_kv or any(layer < 0 for layer in self.layer_kv):
            raise ValueError("soft-gated layer K/V must contain non-negative layers")
        if block_gates.ndim != 2:
            raise ValueError("block_gates must have shape [batch, blocks]")
        self.block_token_counts = tuple(int(value) for value in block_token_counts)
        if (
            len(self.block_token_counts) != block_gates.shape[1]
            or any(value <= 0 for value in self.block_token_counts)
        ):
            raise ValueError("block token counts must be positive and align with gates")
        if not 0 < gate_epsilon < 1:
            raise ValueError("gate_epsilon must be in (0, 1)")
        self.block_gates = block_gates
        self.gate_epsilon = float(gate_epsilon)
        token_gates = torch.repeat_interleave(
            block_gates,
            torch.tensor(self.block_token_counts, device=block_gates.device),
            dim=1,
        )
        # Map [0, 1] to [epsilon, 1] before taking the logarithm.  Unlike a
        # hard clamp this preserves a finite gradient even for gates that have
        # saturated close to zero, while a gate of exactly one remains a true
        # no-op on the attention logits.
        safe_token_gates = (
            token_gates * (1.0 - self.gate_epsilon) + self.gate_epsilon
        )
        self.token_log_gates = safe_token_gates.log()
        self.layer_calls: dict[int, int] = {}

    @staticmethod
    def physical_source_layer(module) -> int:
        return Gemma4StaticKVController.physical_source_layer(module)

    def attend(
        self,
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float,
        scaling: float | None,
        softcap: float | None,
    ):
        source_layer = self.physical_source_layer(module)
        historical = self.layer_kv.get(source_layer)
        if historical is None:
            return eager_attention_reference(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                softcap=softcap,
            )
        historical_key, historical_value = historical
        if historical_key.ndim != 4 or historical_value.shape != historical_key.shape:
            raise ValueError("historical K/V must have shape [batch, heads, tokens, dim]")
        if (
            historical_key.shape[0] != key.shape[0]
            or historical_key.shape[1] != key.shape[1]
            or historical_key.shape[3] != key.shape[3]
        ):
            raise ValueError(f"historical K/V is incompatible with layer {source_layer}")
        if historical_key.shape[2] != self.token_log_gates.shape[1]:
            raise ValueError("historical K/V length does not match expanded block gates")
        if self.token_log_gates.shape[0] != query.shape[0]:
            raise ValueError("gate and attention batch dimensions do not match")

        historical_key = historical_key.to(device=key.device, dtype=key.dtype)
        historical_value = historical_value.to(device=value.device, dtype=value.dtype)
        augmented_key = torch.cat((historical_key, key), dim=2)
        augmented_value = torch.cat((historical_value, value), dim=2)
        augmented_mask = _prepend_native_mask(
            attention_mask,
            query=query,
            historical_length=historical_key.shape[2],
            native_key_length=key.shape[2],
        )

        if scaling is None:
            scaling = module.head_dim**-0.5
        key_states = _repeat_kv(augmented_key, module.num_key_value_groups)
        value_states = _repeat_kv(augmented_value, module.num_key_value_groups)
        weights = torch.matmul(query, key_states.transpose(2, 3)).float() * scaling
        if softcap is not None:
            weights = torch.tanh(weights / softcap) * softcap
        weights = weights + augmented_mask.float()
        historical_length = historical_key.shape[2]
        weights = torch.cat(
            (
                weights[..., :historical_length]
                + self.token_log_gates.to(weights.device).view(
                    query.shape[0], 1, 1, historical_length
                ),
                weights[..., historical_length:],
            ),
            dim=-1,
        )
        probabilities = F.softmax(weights, dim=-1, dtype=torch.float32)
        probabilities = F.dropout(probabilities, p=dropout, training=module.training)
        output = torch.matmul(probabilities.to(value_states.dtype), value_states)
        layer_index = int(module.layer_idx)
        self.layer_calls[layer_index] = self.layer_calls.get(layer_index, 0) + 1
        return output.transpose(1, 2).contiguous(), probabilities.to(query.dtype)


def _registered_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **_kwargs,
):
    with _CONTROLLER_LOCK:
        controller = _CONTROLLERS.get(module)
    if controller is None:
        return eager_attention_reference(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            softcap=softcap,
        )
    return controller.attend(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        softcap=softcap,
    )


def register_attention_backend() -> None:
    global _BACKEND_REGISTERED
    if _BACKEND_REGISTERED:
        return
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(
        ATTENTION_BACKEND_NAME, _registered_attention_forward
    )
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_BACKEND_NAME, eager_mask)
    _BACKEND_REGISTERED = True


def _gemma4_text_attention_modules(model) -> list[Any]:
    modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "Gemma4TextAttention"
    ]
    if not modules:
        raise ValueError("no Gemma4TextAttention modules were found")
    return modules


class Gemma4SoftBlockGateAdapter:
    """Temporarily inject differentiably gated historical K/V into Gemma 4."""

    def __init__(
        self,
        model,
        layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
        block_gates: torch.Tensor,
        block_token_counts: Sequence[int],
        *,
        gate_epsilon: float = 1e-6,
    ) -> None:
        self.model = model
        self.controller = Gemma4SoftBlockGateController(
            layer_kv,
            block_gates,
            block_token_counts,
            gate_epsilon=gate_epsilon,
        )
        self.modules = [
            module
            for module in _gemma4_text_attention_modules(model)
            if self.controller.physical_source_layer(module)
            in self.controller.layer_kv
        ]
        self._saved_implementations: dict[int, tuple[Any, Any]] = {}
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("Gemma4SoftBlockGateAdapter cannot be entered twice")
        register_attention_backend()
        with _CONTROLLER_LOCK:
            for module in self.modules:
                config = module.config
                config_id = id(config)
                if config_id not in self._saved_implementations:
                    self._saved_implementations[config_id] = (
                        config,
                        config._attn_implementation,
                    )
                _CONTROLLERS[module] = self.controller
                config._attn_implementation = ATTENTION_BACKEND_NAME
        self._active = True
        return self.controller

    def __exit__(self, exc_type, exc, traceback):
        with _CONTROLLER_LOCK:
            for module in self.modules:
                _CONTROLLERS.pop(module, None)
            for config, implementation in self._saved_implementations.values():
                config._attn_implementation = implementation
        self._saved_implementations.clear()
        self._active = False
        return False


__all__ = [
    "Gemma4SoftBlockGateAdapter",
    "Gemma4SoftBlockGateController",
    "register_attention_backend",
]
