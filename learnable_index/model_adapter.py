from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


SUPPORTED_MODEL_TYPES = {"gemma4", "gemma4_text"}


def _dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def text_config(model):
    config = model.config
    return config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config


def text_model(model):
    model_type = getattr(model.config, "model_type", None)
    if model_type == "gemma4_text":
        return model.model
    if model_type == "gemma4":
        return model.model.language_model
    raise ValueError(f"learnable_index currently supports Gemma 4 text backbones, found {model_type!r}")


def model_fingerprint(model, model_name: str) -> dict[str, Any]:
    config = text_config(model)
    return {
        "model_name": model_name,
        "model_type": getattr(model.config, "model_type", None),
        "text_model_type": getattr(config, "model_type", None),
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_kv_shared_layers": int(getattr(config, "num_kv_shared_layers", 0)),
        "layer_types": list(getattr(config, "layer_types", [])),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "sliding_window": int(getattr(config, "sliding_window", 0) or 0),
        "transformers_minimum_compatible": "5.12.1",
    }


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    model_name: str
    input_device: torch.device
    fingerprint: dict[str, Any]

    @property
    def text_config(self):
        return text_config(self.model)

    @property
    def text_model(self):
        return text_model(self.model)

    @property
    def physical_cache_layer_count(self) -> int:
        return int(self.text_config.num_hidden_layers) - int(
            getattr(self.text_config, "num_kv_shared_layers", 0)
        )

    @property
    def cache_layer_devices(self) -> tuple[torch.device, ...]:
        layers = getattr(self.text_model, "layers", None)
        if layers is None:
            return (self.input_device,) * self.physical_cache_layer_count
        devices: list[torch.device] = []
        for layer in layers[: self.physical_cache_layer_count]:
            attention = layer.self_attn
            parameter = next(attention.parameters())
            devices.append(parameter.device)
        if len(devices) != self.physical_cache_layer_count:
            raise RuntimeError("could not resolve every physical cache layer device")
        return tuple(devices)


def load_frozen_gemma(
    model_name: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = True,
    cache_dir: str | None = None,
) -> ModelBundle:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment failure
        raise RuntimeError("transformers is required for model collection and replay") from error

    load_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": False,
        "dtype": _dtype_from_name(dtype),
        "attn_implementation": "eager",
    }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device != "auto":
        model.to(torch.device(device))
    model.eval()
    model.requires_grad_(False)
    model_type = getattr(model.config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported model_type {model_type!r}; expected one of {sorted(SUPPORTED_MODEL_TYPES)}")
    config = text_config(model)
    if getattr(config, "_attn_implementation", None) != "eager":
        raise RuntimeError("teacher/replay collection requires Gemma 4 eager attention")
    input_device = text_model(model).embed_tokens.weight.device
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        input_device=input_device,
        fingerprint=model_fingerprint(model, model_name),
    )


def hidden_state_at_layer(output, residual_layer: int) -> torch.Tensor:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("model output did not include hidden states")
    if residual_layer == -1:
        selected = hidden_states[-1]
    else:
        index = residual_layer + 1
        if index <= 0 or index >= len(hidden_states):
            raise IndexError(
                f"residual_layer={residual_layer} is outside {len(hidden_states) - 1} decoder layers"
            )
        selected = hidden_states[index]
    if selected.ndim != 3 or selected.shape[0] != 1:
        raise RuntimeError("expected hidden state shape [1, tokens, hidden]")
    return selected


def new_full_dynamic_cache():
    from transformers.cache_utils import DynamicCache

    # Deliberately omit model config: sparse prefixes must not be truncated by
    # DynamicSlidingWindowLayer while being materialized for explicit masks.
    return DynamicCache()


def cache_from_layer_kv(layer_kv: Iterable[tuple[torch.Tensor, torch.Tensor]]):
    from transformers.cache_utils import DynamicCache

    pairs = tuple((key, value) for key, value in layer_kv)
    return DynamicCache(ddp_cache_data=pairs)


def cache_suffix(cache, maximum_length: int):
    """Return a dynamic cache containing only the newest physical K/V tokens.

    Transformers' ``DynamicCache.crop`` retains the *prefix*.  A strict rolling
    local context needs the opposite operation, so this helper creates a new
    cache from suffix views without copying the underlying K/V tensors.
    """

    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    pairs = layer_kv_from_cache(cache)
    length = int(pairs[0][0].shape[2])
    if length <= maximum_length:
        return cache
    return cache_from_layer_kv(
        (key[:, :, -maximum_length:, :], value[:, :, -maximum_length:, :])
        for key, value in pairs
    )


def layer_kv_from_cache(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, layer in enumerate(cache.layers):
        if not getattr(layer, "is_initialized", False):
            raise RuntimeError(f"cache layer {layer_index} is not initialized")
        key, value = layer.keys, layer.values
        if key.ndim != 4 or value.shape != key.shape:
            raise RuntimeError(f"cache layer {layer_index} has invalid K/V shapes")
        pairs.append((key, value))
    if not pairs:
        raise RuntimeError("cache contains no physical layers")
    lengths = {int(key.shape[2]) for key, _ in pairs}
    if len(lengths) != 1:
        raise RuntimeError(f"cache physical layers have inconsistent lengths: {sorted(lengths)}")
    return tuple(pairs)


def extract_cache_token_range(
    cache,
    *,
    input_length: int,
    start_index: int,
    end_index: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if not 0 <= start_index < end_index <= input_length:
        raise ValueError("cache token range must lie inside the forwarded input")
    expected = end_index - start_index
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, (key, value) in enumerate(layer_kv_from_cache(cache)):
        physical_length = int(key.shape[2])
        physical_offset = input_length - physical_length
        physical_start = start_index - physical_offset
        physical_end = end_index - physical_offset
        if physical_start < 0 or physical_end > physical_length:
            raise RuntimeError(
                f"requested block is no longer present in cache layer {layer_index}: "
                f"input_length={input_length}, physical_length={physical_length}"
            )
        sliced_key = key[:, :, physical_start:physical_end, :].detach().cpu()
        sliced_value = value[:, :, physical_start:physical_end, :].detach().cpu()
        if sliced_key.shape[2] != expected:
            raise RuntimeError("cache slice length mismatch")
        result.append((sliced_key, sliced_value))
    return tuple(result)


def concatenate_layer_kv(
    *parts: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    non_empty = [part for part in parts if part]
    if not non_empty:
        return ()
    layer_count = len(non_empty[0])
    if any(len(part) != layer_count for part in non_empty):
        raise ValueError("cache parts have different physical layer counts")
    return tuple(
        (
            torch.cat([part[layer_index][0] for part in non_empty], dim=2),
            torch.cat([part[layer_index][1] for part in non_empty], dim=2),
        )
        for layer_index in range(layer_count)
    )


def trim_prefix_and_local_kv(
    layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    *,
    prefix_length: int,
    maximum_local_tokens: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if prefix_length < 0 or maximum_local_tokens < 0:
        raise ValueError("prefix and local token limits must be non-negative")
    trimmed: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, (key, value) in enumerate(layer_kv):
        length = int(key.shape[2])
        if value.shape[2] != length or prefix_length > length:
            raise ValueError(f"invalid cache layer {layer_index} for prefix length")
        local_length = length - prefix_length
        local_start = prefix_length + max(0, local_length - maximum_local_tokens)
        keep_key = torch.cat((key[:, :, :prefix_length, :], key[:, :, local_start:, :]), dim=2)
        keep_value = torch.cat((value[:, :, :prefix_length, :], value[:, :, local_start:, :]), dim=2)
        trimmed.append((keep_key, keep_value))
    return tuple(trimmed)


def build_sparse_prefix_mask(
    bundle: ModelBundle,
    *,
    query_length: int,
    prefix_length: int,
    local_past_length: int,
) -> dict[str, torch.Tensor]:
    """Mask `[selected prefix] + [local past] + [current query tokens]`."""

    if query_length <= 0 or prefix_length < 0 or local_past_length < 0:
        raise ValueError("invalid sparse-prefix mask dimensions")
    embedding = bundle.text_model.embed_tokens.weight
    dtype = embedding.dtype if torch.is_floating_point(embedding) else torch.float32
    device = bundle.input_device
    current_length = local_past_length + query_length
    total_key_length = prefix_length + current_length
    mask = torch.zeros((1, 1, query_length, total_key_length), dtype=dtype, device=device)
    key_positions = torch.arange(current_length, device=device)
    query_positions = torch.arange(
        local_past_length,
        local_past_length + query_length,
        device=device,
    )
    disallowed = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    if disallowed.any():
        mask[:, :, :, prefix_length:] = mask[:, :, :, prefix_length:].masked_fill(
            disallowed.view(1, 1, query_length, current_length),
            torch.finfo(dtype).min,
        )
    layer_types = set(getattr(bundle.text_config, "layer_types", ["full_attention"]))
    return {layer_type: mask for layer_type in layer_types}


def build_rolling_local_mask(
    bundle: ModelBundle,
    *,
    past_positions: Iterable[int],
    query_positions: Iterable[int],
    local_context_length: int,
):
    """Build an exact logical-position mask for a chunked rolling local cache.

    ``local_context_length`` counts the current query token, matching a direct
    forward over the final ``L`` tokens.  Batched query tokens therefore see
    neither future tokens in their chunk nor native keys older than ``L - 1``.
    """

    if local_context_length <= 0:
        raise ValueError("local_context_length must be positive")
    past = tuple(int(value) for value in past_positions)
    query = tuple(int(value) for value in query_positions)
    if not query:
        raise ValueError("query_positions must be non-empty")
    combined = past + query
    if any(right != left + 1 for left, right in zip(combined, combined[1:])):
        raise ValueError("rolling cache positions must be contiguous")
    embedding = bundle.text_model.embed_tokens.weight
    dtype = embedding.dtype if torch.is_floating_point(embedding) else torch.float32
    device = bundle.input_device
    key_positions = torch.tensor(combined, device=device, dtype=torch.long)
    query_tensor = torch.tensor(query, device=device, dtype=torch.long)
    allowed = (
        (key_positions[None, :] <= query_tensor[:, None])
        & (key_positions[None, :] > query_tensor[:, None] - local_context_length)
    )
    minimum = torch.finfo(dtype).min
    mask = torch.zeros(
        (1, 1, len(query), len(combined)), device=device, dtype=dtype
    ).masked_fill(~allowed.view(1, 1, len(query), len(combined)), minimum)
    layer_types = set(getattr(bundle.text_config, "layer_types", ["full_attention"]))
    return {layer_type: mask for layer_type in layer_types}


def forward_tokens(
    bundle: ModelBundle,
    token_ids: Iterable[int],
    logical_positions: Iterable[int],
    *,
    past_key_values=None,
    attention_mask=None,
    use_cache: bool,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
    logical_cache_position: bool = False,
):
    token_list = list(token_ids)
    position_list = list(logical_positions)
    if not token_list or len(token_list) != len(position_list):
        raise ValueError("token ids and logical positions must be aligned and non-empty")
    input_ids = torch.tensor([token_list], dtype=torch.long, device=bundle.input_device)
    position_ids = torch.tensor([position_list], dtype=torch.long, device=bundle.input_device)
    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "return_dict": True,
        "output_hidden_states": output_hidden_states,
        "output_attentions": output_attentions,
    }
    if logical_cache_position:
        kwargs["cache_position"] = position_ids[0]
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    with torch.inference_mode():
        return bundle.model(**kwargs)
