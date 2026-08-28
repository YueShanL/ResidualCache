"""Exact layer-online memory injection for Gemma 4 text attention.

The adapter registers a temporary Hugging Face attention backend.  At every
selected decoder layer it:

1. retrieves historical K/V from the layer's independent token memory;
2. prepends those K/V tensors to the current attention operation; and
3. writes the current layer K/V only after the attention output is computed.

No residual-index code is imported, and no second model pass is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading
import weakref
from typing import Any, Iterable

from .torch_token_memory import TokenMemoryConfig, create_token_memory_bank


ATTENTION_BACKEND_NAME = "residual_cache_token_memory"
_CONTROLLERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_CONTROLLER_LOCK = threading.RLock()
_BACKEND_REGISTERED = False


def _require_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - exercised by the CLI.
        raise RuntimeError(
            "Gemma 4 memory evaluation requires torch from the project environment."
        ) from exc
    return torch, functional


def _repeat_kv(hidden_states, repetitions: int):
    batch, key_value_heads, tokens, head_dim = hidden_states.shape
    if repetitions == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, key_value_heads, repetitions, tokens, head_dim
    )
    return expanded.reshape(
        batch, key_value_heads * repetitions, tokens, head_dim
    )


def eager_attention_reference(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
):
    """Gemma-compatible eager attention used by both baseline and adapter."""

    torch, functional = _require_torch()
    if scaling is None:
        scaling = module.head_dim**-0.5
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        weights = torch.tanh(weights / softcap) * softcap
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value_states)
    return output.transpose(1, 2).contiguous(), weights


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
    """Register the adapter backend once in Transformers' global interface."""

    global _BACKEND_REGISTERED
    if _BACKEND_REGISTERED:
        return
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.masking_utils import (
            ALL_MASK_ATTENTION_FUNCTIONS,
            eager_mask,
        )
    except ImportError as exc:  # pragma: no cover - exercised by the CLI.
        raise RuntimeError(
            "The installed Transformers version lacks the attention interface."
        ) from exc
    ALL_ATTENTION_FUNCTIONS.register(
        ATTENTION_BACKEND_NAME, _registered_attention_forward
    )
    # Transformers otherwise treats an unknown custom backend like an external
    # runtime and returns no mask. That silently turns the decoder
    # bidirectional and produces dramatically invalid perplexity.
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_BACKEND_NAME, eager_mask)
    _BACKEND_REGISTERED = True


@dataclass(frozen=True)
class Gemma4MemoryAdapterConfig:
    """Runtime options which do not alter the memory policy itself."""

    memory_device: str | None = None
    augmented_layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.augmented_layers is not None:
            if not self.augmented_layers:
                raise ValueError("augmented_layers cannot be empty.")
            if min(self.augmented_layers) < 0:
                raise ValueError("Layer indices must be non-negative.")
            if len(set(self.augmented_layers)) != len(self.augmented_layers):
                raise ValueError("augmented_layers cannot contain duplicates.")


class Gemma4MemoryController:
    """Owns independent per-layer memory banks for one model."""

    def __init__(
        self,
        memory_config: TokenMemoryConfig,
        adapter_config: Gemma4MemoryAdapterConfig | None = None,
    ):
        self.memory_config = memory_config
        self.adapter_config = adapter_config or Gemma4MemoryAdapterConfig()
        self.banks: dict[int, Any] = {}
        self.layer_calls: dict[int, int] = {}
        self.retrieved_tokens = 0
        self.written_tokens = 0

    def reset(self) -> None:
        self.banks.clear()
        self.layer_calls.clear()
        self.retrieved_tokens = 0
        self.written_tokens = 0

    def _bank(self, module, key):
        layer_index = int(module.layer_idx)
        bank = self.banks.get(layer_index)
        if bank is None:
            device = (
                key.device
                if self.adapter_config.memory_device is None
                else self.adapter_config.memory_device
            )
            bank = create_token_memory_bank(
                batch_size=key.shape[0],
                kv_heads=key.shape[1],
                head_dim=key.shape[3],
                device=device,
                dtype=key.dtype,
                config=self.memory_config,
            )
            self.banks[layer_index] = bank
        elif bank.batch_size != key.shape[0]:
            raise ValueError(
                "Batch size changed while memory was populated. Reset the controller "
                "between differently batched streams."
            )
        return bank

    @staticmethod
    def _prepend_mask(attention_mask, valid, *, query, native_key_length: int):
        torch, _functional = _require_torch()
        prefix_length = valid.shape[1]
        if prefix_length == 0:
            return attention_mask
        if attention_mask is None:
            prefix = torch.zeros(
                (query.shape[0], 1, query.shape[2], prefix_length),
                device=query.device,
                dtype=query.dtype,
            )
            native = torch.zeros(
                (query.shape[0], 1, query.shape[2], native_key_length),
                device=query.device,
                dtype=query.dtype,
            )
            minimum = torch.finfo(query.dtype).min
            prefix = prefix.masked_fill(~valid[:, None, None, :], minimum)
            return torch.cat((prefix, native), dim=-1)
        if attention_mask.ndim != 4:
            raise ValueError("Gemma eager attention requires a four-dimensional mask.")
        if attention_mask.shape[-1] != native_key_length:
            raise ValueError(
                "Memory evaluation requires a mask matching the native K/V length."
            )
        prefix = torch.zeros(
            (
                attention_mask.shape[0],
                attention_mask.shape[1],
                attention_mask.shape[2],
                prefix_length,
            ),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        invalid = ~valid[:, None, None, :]
        minimum = torch.finfo(attention_mask.dtype).min
        prefix = prefix.masked_fill(invalid, minimum)
        return torch.cat((prefix, attention_mask), dim=-1)

    def attend(
        self,
        module,
        query,
        key,
        value,
        attention_mask,
        *,
        dropout: float,
        scaling: float | None,
        softcap: float | None,
    ):
        torch, _functional = _require_torch()
        if key.shape[2] != query.shape[2]:
            raise ValueError(
                "The standalone CAMELoT protocol requires use_cache=False; "
                "native K/V must contain exactly the current window."
            )
        bank = self._bank(module, key)
        if hasattr(bank, "keys"):
            bank_device = bank.keys.device
        elif hasattr(bank, "record_keys"):
            bank_device = bank.record_keys.device
        else:
            bank_device = bank.device
        bank_keys = key.detach().to(bank_device)
        bank_values = value.detach().to(bank_device)
        memory_keys, memory_values, valid = bank.retrieve(bank_keys)
        has_memory = bool(valid.any().item())
        if has_memory:
            memory_keys = memory_keys.to(device=key.device, dtype=key.dtype)
            memory_values = memory_values.to(device=value.device, dtype=value.dtype)
            valid = valid.to(query.device)
            augmented_key = torch.cat((memory_keys, key), dim=2)
            augmented_value = torch.cat((memory_values, value), dim=2)
            augmented_mask = self._prepend_mask(
                attention_mask,
                valid,
                query=query,
                native_key_length=key.shape[2],
            )
            self.retrieved_tokens += int(valid.sum().item())
        else:
            augmented_key = key
            augmented_value = value
            augmented_mask = attention_mask
        output = eager_attention_reference(
            module,
            query,
            augmented_key,
            augmented_value,
            augmented_mask,
            dropout=dropout,
            scaling=scaling,
            softcap=softcap,
        )
        bank.write(bank_keys, bank_values)
        layer_index = int(module.layer_idx)
        self.layer_calls[layer_index] = self.layer_calls.get(layer_index, 0) + 1
        self.written_tokens += key.shape[0] * key.shape[2]
        return output

    @property
    def memory_bytes(self) -> int:
        return sum(bank.memory_bytes for bank in self.banks.values())

    def snapshot(self) -> dict[str, Any]:
        return {
            "memory_config": asdict(self.memory_config),
            "adapter_config": asdict(self.adapter_config),
            "layer_count": len(self.banks),
            "layer_calls": dict(sorted(self.layer_calls.items())),
            "retrieved_tokens": self.retrieved_tokens,
            "written_tokens": self.written_tokens,
            "memory_bytes": self.memory_bytes,
            "layers": {
                str(index): bank.snapshot()
                for index, bank in sorted(self.banks.items())
            },
        }


class Gemma4StaticKVController:
    """Inject a preselected, variable-length historical K/V view per layer.

    Unlike :class:`Gemma4MemoryController`, this controller neither retrieves
    from nor writes to a memory bank.  It is the narrow execution boundary used
    after an external policy has selected complete memory clusters.  Gemma 4's
    KV-sharing decoder layers are mapped back to the final physical cache layer
    of the same attention type, matching the model's native ``shared_kv_states``
    behavior.
    """

    def __init__(
        self,
        layer_kv: dict[int, tuple[Any, Any]],
        *,
        collect_historical_usage: bool = False,
    ):
        self.layer_kv = {int(layer): pair for layer, pair in layer_kv.items()}
        if any(layer < 0 for layer in self.layer_kv):
            raise ValueError("static K/V layer indices must be non-negative")
        self.layer_calls: dict[int, int] = {}
        self.retrieved_tokens_by_layer: dict[int, int] = {}
        self.collect_historical_usage = bool(collect_historical_usage)
        self._historical_usage_sums: dict[int, Any] = {}
        self._historical_usage_denominators: dict[int, int] = {}

    @staticmethod
    def physical_source_layer(module) -> int:
        layer_index = int(module.layer_idx)
        if not bool(getattr(module, "is_kv_shared_layer", False)):
            return layer_index
        first_shared = int(module.config.num_hidden_layers) - int(
            getattr(module.config, "num_kv_shared_layers", 0)
        )
        previous_types = list(module.config.layer_types[:first_shared])
        try:
            reverse_offset = previous_types[::-1].index(module.layer_type)
        except ValueError as error:
            raise ValueError(
                f"no physical source layer exists for attention type {module.layer_type!r}"
            ) from error
        return len(previous_types) - 1 - reverse_offset

    def attend(
        self,
        module,
        query,
        key,
        value,
        attention_mask,
        *,
        dropout: float,
        scaling: float | None,
        softcap: float | None,
    ):
        torch, _functional = _require_torch()
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
            raise ValueError("static historical K/V must have shape [batch, heads, tokens, dim]")
        if (
            historical_key.shape[0] != key.shape[0]
            or historical_key.shape[1] != key.shape[1]
            or historical_key.shape[3] != key.shape[3]
        ):
            raise ValueError(
                f"static historical K/V is incompatible with layer {source_layer}"
            )
        historical_key = historical_key.to(device=key.device, dtype=key.dtype)
        historical_value = historical_value.to(device=value.device, dtype=value.dtype)
        valid = torch.ones(
            (key.shape[0], historical_key.shape[2]),
            dtype=torch.bool,
            device=query.device,
        )
        augmented_mask = Gemma4MemoryController._prepend_mask(
            attention_mask,
            valid,
            query=query,
            native_key_length=key.shape[2],
        )
        output = eager_attention_reference(
            module,
            query,
            torch.cat((historical_key, key), dim=2),
            torch.cat((historical_value, value), dim=2),
            augmented_mask,
            dropout=dropout,
            scaling=scaling,
            softcap=softcap,
        )
        layer_index = int(module.layer_idx)
        self.layer_calls[layer_index] = self.layer_calls.get(layer_index, 0) + 1
        self.retrieved_tokens_by_layer[source_layer] = (
            self.retrieved_tokens_by_layer.get(source_layer, 0)
            + int(historical_key.shape[0] * historical_key.shape[2])
        )
        if self.collect_historical_usage:
            weights = output[1][..., : historical_key.shape[2]].detach().float()
            usage_sum = weights.sum(dim=(0, 1, 2))
            denominator = int(weights.shape[0] * weights.shape[1] * weights.shape[2])
            previous = self._historical_usage_sums.get(source_layer)
            if previous is None:
                self._historical_usage_sums[source_layer] = usage_sum
            else:
                if previous.shape != usage_sum.shape:
                    raise ValueError(
                        "historical K/V length changed while collecting recall usage"
                    )
                previous.add_(usage_sum)
            self._historical_usage_denominators[source_layer] = (
                self._historical_usage_denominators.get(source_layer, 0) + denominator
            )
        return output

    def historical_usage_rates(self) -> dict[int, Any]:
        """Return mean attention probability aligned with each historical K/V row."""

        return {
            layer: values / self._historical_usage_denominators[layer]
            for layer, values in self._historical_usage_sums.items()
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "layer_calls": dict(sorted(self.layer_calls.items())),
            "retrieved_tokens_by_physical_layer": dict(
                sorted(self.retrieved_tokens_by_layer.items())
            ),
            "visible_tokens_by_physical_layer": {
                str(layer): int(key.shape[0] * key.shape[2])
                for layer, (key, _value) in sorted(self.layer_kv.items())
            },
            "historical_usage_collected": self.collect_historical_usage,
            "historical_usage_metric": (
                "mean_attention_probability_per_historical_record"
                if self.collect_historical_usage
                else None
            ),
        }


def _gemma4_text_attention_modules(model) -> list[Any]:
    modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "Gemma4TextAttention"
    ]
    if not modules:
        raise ValueError(
            "No Gemma4TextAttention modules were found. This adapter deliberately "
            "supports Gemma 4 only so model changes cannot silently alter the experiment."
        )
    return modules


class Gemma4MemoryAdapter:
    """Context manager that temporarily enables token memory on a Gemma 4 model."""

    def __init__(
        self,
        model,
        memory_config: TokenMemoryConfig,
        adapter_config: Gemma4MemoryAdapterConfig | None = None,
    ):
        self.model = model
        self.controller = Gemma4MemoryController(memory_config, adapter_config)
        self.modules = _gemma4_text_attention_modules(model)
        selected = self.controller.adapter_config.augmented_layers
        if selected is not None:
            selected_set = set(selected)
            known = {int(module.layer_idx) for module in self.modules}
            missing = selected_set - known
            if missing:
                raise ValueError(f"Unknown Gemma 4 layer indices: {sorted(missing)}")
            self.modules = [
                module for module in self.modules if int(module.layer_idx) in selected_set
            ]
        self._saved_implementations: dict[int, tuple[Any, Any]] = {}
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("Gemma4MemoryAdapter cannot be entered twice.")
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


class Gemma4StaticKVAdapter:
    """Temporarily inject externally selected historical K/V into Gemma 4."""

    def __init__(
        self,
        model,
        layer_kv: dict[int, tuple[Any, Any]],
        *,
        collect_historical_usage: bool = False,
    ):
        self.model = model
        self.controller = Gemma4StaticKVController(
            layer_kv,
            collect_historical_usage=collect_historical_usage,
        )
        self.modules = [
            module
            for module in _gemma4_text_attention_modules(model)
            if self.controller.physical_source_layer(module) in self.controller.layer_kv
        ]
        self._saved_implementations: dict[int, tuple[Any, Any]] = {}
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("Gemma4StaticKVAdapter cannot be entered twice.")
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


def parse_augmented_layers(specification: str | None) -> tuple[int, ...] | None:
    if specification is None or specification.strip().lower() in {"", "all"}:
        return None
    values: list[int] = []
    for piece in specification.split(","):
        piece = piece.strip()
        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending layer range {piece!r}.")
            values.extend(range(start, end + 1))
        else:
            values.append(int(piece))
    return tuple(values)


__all__ = [
    "ATTENTION_BACKEND_NAME",
    "Gemma4MemoryAdapter",
    "Gemma4MemoryAdapterConfig",
    "Gemma4MemoryController",
    "Gemma4StaticKVAdapter",
    "Gemma4StaticKVController",
    "eager_attention_reference",
    "parse_augmented_layers",
    "register_attention_backend",
]
