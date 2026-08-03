from __future__ import annotations

from dataclasses import dataclass

from .answer_block_cache import (
    _forward,
    _hidden,
    _transformer_layers,
)


@dataclass(frozen=True)
class ContinuousTurnTokens:
    history_ids: list[int]
    continuation_ids: list[int]
    full_ids: list[int]
    assistant_text: str


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def build_continuous_turn_tokens(
    tokenizer,
    *,
    history_user_content: str,
    history_prompt_ids: list[int],
    history_generated_ids: list[int],
    query_user_content: str,
) -> ContinuousTurnTokens:
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "Strict continuation equivalence requires a chat template."
        )
    assistant_text = tokenizer.decode(
        history_generated_ids,
        skip_special_tokens=True,
    )
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": history_user_content.strip(),
            },
            {
                "role": "assistant",
                "content": assistant_text,
            },
            {
                "role": "user",
                "content": query_user_content.strip(),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_ids = tokenizer(
        rendered,
        add_special_tokens=False,
    ).input_ids
    history_ids = history_prompt_ids + history_generated_ids
    mismatch = _first_mismatch(
        history_ids,
        full_ids[: len(history_ids)],
    )
    if mismatch is not None:
        expected_id = (
            history_ids[mismatch]
            if mismatch < len(history_ids)
            else None
        )
        actual_id = (
            full_ids[mismatch]
            if mismatch < len(full_ids)
            else None
        )
        raise RuntimeError(
            "Generated history is not an exact prefix of the "
            "standard multi-turn chat template: "
            f"mismatch at token {mismatch}, "
            f"history={expected_id}, full={actual_id}."
        )
    continuation_ids = full_ids[len(history_ids) :]
    if not continuation_ids:
        raise RuntimeError(
            "Multi-turn chat template produced no query continuation."
        )
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if (
        bos_token_id is not None
        and continuation_ids[0] == int(bos_token_id)
    ):
        raise RuntimeError(
            "Query continuation incorrectly starts with BOS."
        )
    return ContinuousTurnTokens(
        history_ids=history_ids,
        continuation_ids=continuation_ids,
        full_ids=full_ids,
        assistant_text=assistant_text,
    )


class _LayerStateCapture:
    def __init__(
        self,
        model,
        *,
        token_start: int,
        layer_indexes: tuple[int, ...] | None = None,
    ):
        self.model = model
        self.token_start = token_start
        self.layer_indexes = layer_indexes
        self.states: dict[int, object] = {}
        self.handles = []

    def __enter__(self):
        layers = _transformer_layers(self.model)
        layer_indexes = (
            range(len(layers))
            if self.layer_indexes is None
            else self.layer_indexes
        )
        for layer_index in layer_indexes:
            layer = layers[layer_index]

            def save_state(
                _module,
                _inputs,
                output,
                *,
                captured_layer=layer_index,
            ):
                hidden = _hidden(output)
                self.states[captured_layer] = (
                    hidden[
                        0,
                        self.token_start :,
                        :,
                    ]
                    .detach()
                    .float()
                    .cpu()
                )

            self.handles.append(
                layer.register_forward_hook(save_state)
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class _LayerInputCapture:
    def __init__(
        self,
        model,
        *,
        layer_indexes: tuple[int, ...],
        token_start: int,
    ):
        self.model = model
        self.layer_indexes = layer_indexes
        self.token_start = token_start
        self.hidden_states: dict[int, object] = {}
        self.per_layer_inputs: dict[int, object] = {}
        self.handles = []

    def __enter__(self):
        layers = _transformer_layers(self.model)
        for layer_index in self.layer_indexes:
            layer = layers[layer_index]

            def save_inputs(
                _module,
                args,
                _kwargs,
                *,
                captured_layer=layer_index,
            ):
                self.hidden_states[captured_layer] = (
                    args[0][0, self.token_start :, :]
                    .detach()
                    .float()
                    .cpu()
                )
                if len(args) > 1 and args[1] is not None:
                    self.per_layer_inputs[captured_layer] = (
                        args[1][0, self.token_start :, :]
                        .detach()
                        .float()
                        .cpu()
                    )

            self.handles.append(
                layer.register_forward_pre_hook(
                    save_inputs,
                    with_kwargs=True,
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class _NamedOutputCapture:
    def __init__(
        self,
        modules: dict[str, object],
        *,
        token_start: int,
    ):
        self.modules = modules
        self.token_start = token_start
        self.outputs: dict[str, object] = {}
        self.handles = []

    def __enter__(self):
        for name, module in self.modules.items():
            def save_output(
                _module,
                _args,
                output,
                *,
                captured_name=name,
            ):
                hidden = _hidden(output)
                self.outputs[captured_name] = (
                    hidden[0, self.token_start :, :]
                    .detach()
                    .float()
                    .cpu()
                )

            self.handles.append(
                module.register_forward_hook(save_output)
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class _AttentionMaskCapture:
    def __init__(
        self,
        model,
        *,
        layer_indexes: tuple[int, ...],
    ):
        self.model = model
        self.layer_indexes = layer_indexes
        self.masks: dict[int, object] = {}
        self.handles = []

    def __enter__(self):
        layers = _transformer_layers(self.model)
        for layer_index in self.layer_indexes:
            def save_mask(
                _module,
                _args,
                kwargs,
                *,
                captured_layer=layer_index,
            ):
                mask = kwargs.get("attention_mask")
                self.masks[captured_layer] = (
                    mask.detach().cpu()
                    if mask is not None
                    else None
                )

            self.handles.append(
                layers[layer_index].register_forward_pre_hook(
                    save_mask,
                    with_kwargs=True,
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _mask_metrics(
    torch,
    continuous_mask,
    cached_mask,
    *,
    history_tokens: int,
) -> dict:
    if continuous_mask is None or cached_mask is None:
        return {
            "continuous_is_none": continuous_mask is None,
            "cached_is_none": cached_mask is None,
            "equal": continuous_mask is cached_mask,
        }
    continuous_tail = continuous_mask[
        ...,
        history_tokens:,
        :,
    ]
    if continuous_tail.shape != cached_mask.shape:
        return {
            "continuous_shape": list(
                continuous_tail.shape
            ),
            "cached_shape": list(cached_mask.shape),
            "equal": False,
            "shape_equal": False,
        }
    difference = (
        continuous_tail.float()
        - cached_mask.float()
    )
    finite_difference = difference[
        torch.isfinite(difference)
    ]
    return {
        "continuous_shape": list(
            continuous_tail.shape
        ),
        "cached_shape": list(cached_mask.shape),
        "shape_equal": True,
        "equal": bool(
            torch.equal(
                continuous_tail,
                cached_mask,
            )
        ),
        "different_elements": int(
            (
                continuous_tail
                != cached_mask
            )
            .sum()
            .item()
        ),
        "finite_max_abs_diff": (
            float(finite_difference.abs().max().item())
            if finite_difference.numel()
            else 0.0
        ),
    }


def _state_metrics(torch, reference, candidate) -> dict:
    if reference.shape != candidate.shape:
        raise RuntimeError(
            "Layer state shape mismatch: "
            f"continuous={tuple(reference.shape)}, "
            f"cached={tuple(candidate.shape)}."
        )
    difference = candidate - reference
    reference_rms = float(
        reference.square().mean().sqrt().item()
    )
    difference_rms = float(
        difference.square().mean().sqrt().item()
    )
    token_cosine = torch.nn.functional.cosine_similarity(
        reference,
        candidate,
        dim=-1,
        eps=1e-12,
    )
    token_difference_rms = (
        difference.square()
        .mean(dim=-1)
        .sqrt()
    )
    token_reference_rms = (
        reference.square()
        .mean(dim=-1)
        .sqrt()
    )
    token_relative_rmse = (
        token_difference_rms
        / token_reference_rms.clamp_min(1e-12)
    )
    worst_token_index = int(
        token_relative_rmse.argmax().item()
    )
    last_reference = reference[-1]
    last_candidate = candidate[-1]
    last_difference = last_candidate - last_reference
    last_reference_rms = float(
        last_reference.square().mean().sqrt().item()
    )
    last_difference_rms = float(
        last_difference.square().mean().sqrt().item()
    )
    return {
        "tokens": int(reference.shape[0]),
        "hidden_size": int(reference.shape[1]),
        "max_abs_diff": float(
            difference.abs().max().item()
        ),
        "mean_abs_diff": float(
            difference.abs().mean().item()
        ),
        "reference_rms": reference_rms,
        "difference_rms": difference_rms,
        "relative_rmse": (
            difference_rms / reference_rms
            if reference_rms
            else difference_rms
        ),
        "min_token_cosine": float(
            token_cosine.min().item()
        ),
        "mean_token_cosine": float(
            token_cosine.mean().item()
        ),
        "worst_token_index": worst_token_index,
        "worst_token_relative_rmse": float(
            token_relative_rmse[
                worst_token_index
            ].item()
        ),
        "last_token_max_abs_diff": float(
            last_difference.abs().max().item()
        ),
        "last_token_mean_abs_diff": float(
            last_difference.abs().mean().item()
        ),
        "last_token_relative_rmse": (
            last_difference_rms / last_reference_rms
            if last_reference_rms
            else last_difference_rms
        ),
        "last_token_cosine": float(
            torch.nn.functional.cosine_similarity(
                last_reference.unsqueeze(0),
                last_candidate.unsqueeze(0),
                dim=-1,
                eps=1e-12,
            ).item()
        ),
    }


def _cache_layer_type(model, layer_index: int) -> str | None:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None or layer_index >= len(layer_types):
        return None
    return str(layer_types[layer_index])


def _cache_prefix_metrics(
    torch,
    model,
    continuous_cache,
    history_cache,
    *,
    history_tokens: int,
) -> dict:
    continuous_layers = getattr(continuous_cache, "layers", None)
    history_layers = getattr(history_cache, "layers", None)
    if continuous_layers is None or history_layers is None:
        return {
            "available": False,
            "reason": "Cache objects do not expose physical layers.",
        }
    if len(continuous_layers) != len(history_layers):
        raise RuntimeError(
            "Continuous/history cache layer counts differ: "
            f"{len(continuous_layers)} != {len(history_layers)}."
        )

    report = {}
    for layer_index, (continuous_layer, history_layer) in enumerate(
        zip(continuous_layers, history_layers)
    ):
        continuous_key = continuous_layer.keys
        continuous_value = continuous_layer.values
        history_key = history_layer.keys
        history_value = history_layer.values
        continuous_logical = int(
            continuous_layer.get_seq_length()
        )
        history_logical = int(history_layer.get_seq_length())
        continuous_offset = (
            continuous_logical - int(continuous_key.shape[-2])
        )
        history_offset = (
            history_logical - int(history_key.shape[-2])
        )
        overlap_start = max(continuous_offset, history_offset, 0)
        overlap_end = min(
            history_tokens,
            continuous_logical,
            history_logical,
        )
        layer_report = {
            "layer_type": _cache_layer_type(
                model,
                layer_index,
            ),
            "continuous_logical_tokens": continuous_logical,
            "continuous_physical_tokens": int(
                continuous_key.shape[-2]
            ),
            "continuous_physical_start": continuous_offset,
            "history_logical_tokens": history_logical,
            "history_physical_tokens": int(
                history_key.shape[-2]
            ),
            "history_physical_start": history_offset,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_tokens": max(
                overlap_end - overlap_start,
                0,
            ),
        }
        if overlap_end <= overlap_start:
            layer_report["comparable"] = False
            report[str(layer_index)] = layer_report
            continue

        def token_matrix(tensor, physical_start):
            start = overlap_start - physical_start
            end = overlap_end - physical_start
            return (
                tensor[0, :, start:end, :]
                .permute(1, 0, 2)
                .reshape(overlap_end - overlap_start, -1)
                .detach()
                .float()
                .cpu()
            )

        layer_report.update(
            {
                "comparable": True,
                "key": _state_metrics(
                    torch,
                    token_matrix(
                        continuous_key,
                        continuous_offset,
                    ),
                    token_matrix(
                        history_key,
                        history_offset,
                    ),
                ),
                "value": _state_metrics(
                    torch,
                    token_matrix(
                        continuous_value,
                        continuous_offset,
                    ),
                    token_matrix(
                        history_value,
                        history_offset,
                    ),
                ),
            }
        )
        report[str(layer_index)] = layer_report
    return {
        "available": True,
        "layers": report,
    }


def _diagnostic_layer_indexes(model) -> tuple[int, ...]:
    layer_count = len(_transformer_layers(model))
    preferred = (
        0,
        1,
        5,
        11,
        16,
        17,
        18,
        19,
        22,
        23,
        24,
        29,
        35,
        41,
    )
    return tuple(
        layer_index
        for layer_index in preferred
        if layer_index < layer_count
    )


def _first_layer_stage_modules(model) -> dict[str, object]:
    layer = _transformer_layers(model)[0]
    modules = {
        "input_layernorm": layer.input_layernorm,
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "self_attention_output": layer.self_attn,
        "mlp_output": layer.mlp,
        "per_layer_projection": layer.per_layer_projection,
    }
    if layer.self_attn.v_proj is not None:
        modules["v_proj"] = layer.self_attn.v_proj
    return modules


def _stage_diagnostics(
    torch,
    continuous_outputs: dict[str, object],
    history_outputs: dict[str, object],
    cached_outputs: dict[str, object],
    *,
    history_tokens: int,
) -> dict:
    if set(continuous_outputs) != set(history_outputs):
        raise RuntimeError(
            "Continuous/history stage capture sets differ."
        )
    if set(continuous_outputs) != set(cached_outputs):
        raise RuntimeError(
            "Continuous/cached stage capture sets differ."
        )
    report = {}
    for name in continuous_outputs:
        continuous = continuous_outputs[name]
        report[name] = {
            "history": _state_metrics(
                torch,
                continuous[:history_tokens],
                history_outputs[name],
            ),
            "query": _state_metrics(
                torch,
                continuous[history_tokens:],
                cached_outputs[name],
            ),
        }
    return report


def _q_projection_shape_probe(
    torch,
    model,
    normalized_full_state,
    *,
    history_tokens: int,
) -> dict:
    q_proj = _transformer_layers(model)[0].self_attn.q_proj
    weight = q_proj.weight
    device = weight.device
    normalized_full_state = normalized_full_state.to(device)

    def compare(dtype):
        inputs = normalized_full_state.to(dtype=dtype)
        projection_weight = weight.to(dtype=dtype)
        full = torch.nn.functional.linear(
            inputs,
            projection_weight,
        )
        prefix = torch.nn.functional.linear(
            inputs[:history_tokens],
            projection_weight,
        )
        metrics = _state_metrics(
            torch,
            full[:history_tokens].detach().float().cpu(),
            prefix.detach().float().cpu(),
        )
        del inputs, projection_weight, full, prefix
        return metrics

    with torch.no_grad():
        native = compare(weight.dtype)
        tf32_was_enabled = None
        if device.type == "cuda":
            tf32_was_enabled = (
                torch.backends.cuda.matmul.allow_tf32
            )
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            float32 = compare(torch.float32)
        finally:
            if tf32_was_enabled is not None:
                torch.backends.cuda.matmul.allow_tf32 = (
                    tf32_was_enabled
                )
    return {
        "operation": "layer_0_q_proj",
        "full_rows": int(normalized_full_state.shape[0]),
        "prefix_rows": history_tokens,
        "native_dtype": str(weight.dtype),
        "native": native,
        "float32_tf32_disabled": float32,
    }


def validate_continuation_equivalence(
    torch,
    model,
    tokens: ContinuousTurnTokens,
    *,
    start_position: int = 0,
    relative_rmse_tolerance: float = 5e-3,
    min_token_cosine_tolerance: float = 0.9999,
    return_continuous_output: bool = False,
):
    if start_position < 0:
        raise ValueError("start_position must be non-negative.")
    device = next(model.parameters()).device
    full_input = torch.tensor(
        [tokens.full_ids],
        dtype=torch.long,
        device=device,
    )
    history_input = torch.tensor(
        [tokens.history_ids],
        dtype=torch.long,
        device=device,
    )
    continuation_input = torch.tensor(
        [tokens.continuation_ids],
        dtype=torch.long,
        device=device,
    )

    mask_layer_indexes = (
        0,
        min(5, len(_transformer_layers(model)) - 1),
    )
    diagnostic_layer_indexes = _diagnostic_layer_indexes(model)
    first_layer_stage_modules = _first_layer_stage_modules(
        model
    )
    with (
        _LayerStateCapture(
            model,
            token_start=len(tokens.history_ids),
        ) as continuous_capture,
        _LayerStateCapture(
            model,
            token_start=0,
            layer_indexes=diagnostic_layer_indexes,
        ) as continuous_full_capture,
        _LayerInputCapture(
            model,
            layer_indexes=diagnostic_layer_indexes,
            token_start=0,
        ) as continuous_input_capture,
        _NamedOutputCapture(
            first_layer_stage_modules,
            token_start=0,
        ) as continuous_stage_capture,
        _AttentionMaskCapture(
            model,
            layer_indexes=mask_layer_indexes,
        ) as continuous_mask_capture,
    ):
        continuous_output, _ = _forward(
            torch,
            model,
            full_input,
            start_position,
        )

    with (
        _LayerStateCapture(
            model,
            token_start=0,
            layer_indexes=diagnostic_layer_indexes,
        ) as history_capture,
        _LayerInputCapture(
            model,
            layer_indexes=diagnostic_layer_indexes,
            token_start=0,
        ) as history_input_capture,
        _NamedOutputCapture(
            first_layer_stage_modules,
            token_start=0,
        ) as history_stage_capture,
    ):
        history_output, _ = _forward(
            torch,
            model,
            history_input,
            start_position,
        )
    history_cache_report = _cache_prefix_metrics(
        torch,
        model,
        continuous_output.past_key_values,
        history_output.past_key_values,
        history_tokens=len(tokens.history_ids),
    )
    with (
        _LayerStateCapture(
            model,
            token_start=0,
        ) as cached_capture,
        _LayerInputCapture(
            model,
            layer_indexes=diagnostic_layer_indexes,
            token_start=0,
        ) as cached_input_capture,
        _NamedOutputCapture(
            first_layer_stage_modules,
            token_start=0,
        ) as cached_stage_capture,
        _AttentionMaskCapture(
            model,
            layer_indexes=mask_layer_indexes,
        ) as cached_mask_capture,
    ):
        cached_output, _ = _forward(
            torch,
            model,
            continuation_input,
            start_position + len(tokens.history_ids),
            past_key_values=history_output.past_key_values,
        )

    layer_indexes = sorted(continuous_capture.states)
    if layer_indexes != sorted(cached_capture.states):
        raise RuntimeError(
            "Continuous/cached layer capture sets differ."
        )
    layers = {}
    for layer_index in layer_indexes:
        layers[str(layer_index)] = _state_metrics(
            torch,
            continuous_capture.states[layer_index],
            cached_capture.states[layer_index],
        )

    history_tokens = len(tokens.history_ids)
    q_projection_shape_probe = _q_projection_shape_probe(
        torch,
        model,
        continuous_stage_capture.outputs[
            "input_layernorm"
        ],
        history_tokens=history_tokens,
    )
    boundary_diagnostics = {}
    for layer_index in diagnostic_layer_indexes:
        continuous_full_state = (
            continuous_full_capture.states[layer_index]
        )
        layer_report = {
            "layer_type": _cache_layer_type(
                model,
                layer_index,
            ),
            "history_hidden_input": _state_metrics(
                torch,
                continuous_input_capture.hidden_states[
                    layer_index
                ][:history_tokens],
                history_input_capture.hidden_states[
                    layer_index
                ],
            ),
            "history_layer_output": _state_metrics(
                torch,
                continuous_full_state[:history_tokens],
                history_capture.states[layer_index],
            ),
            "query_hidden_input": _state_metrics(
                torch,
                continuous_input_capture.hidden_states[
                    layer_index
                ][history_tokens:],
                cached_input_capture.hidden_states[
                    layer_index
                ],
            ),
            "query_layer_output": layers[str(layer_index)],
        }
        continuous_ple = (
            continuous_input_capture.per_layer_inputs.get(
                layer_index
            )
        )
        history_ple = (
            history_input_capture.per_layer_inputs.get(
                layer_index
            )
        )
        cached_ple = (
            cached_input_capture.per_layer_inputs.get(
                layer_index
            )
        )
        if (
            continuous_ple is not None
            and history_ple is not None
            and cached_ple is not None
        ):
            layer_report["history_per_layer_input"] = (
                _state_metrics(
                    torch,
                    continuous_ple[:history_tokens],
                    history_ple,
                )
            )
            layer_report["query_per_layer_input"] = (
                _state_metrics(
                    torch,
                    continuous_ple[history_tokens:],
                    cached_ple,
                )
            )
        boundary_diagnostics[str(layer_index)] = layer_report

    continuous_logits = (
        continuous_output.logits[
            :,
            len(tokens.history_ids) :,
            :,
        ]
        .detach()
        .float()
        .cpu()
    )
    cached_logits = (
        cached_output.logits.detach().float().cpu()
    )
    logits_metrics = _state_metrics(
        torch,
        continuous_logits[0],
        cached_logits[0],
    )
    continuous_next_token = int(
        continuous_logits[:, -1, :]
        .argmax(dim=-1)
        .item()
    )
    cached_next_token = int(
        cached_logits[:, -1, :]
        .argmax(dim=-1)
        .item()
    )
    max_layer_relative_rmse = max(
        metrics["relative_rmse"]
        for metrics in layers.values()
    )
    min_layer_token_cosine = min(
        metrics["min_token_cosine"]
        for metrics in layers.values()
    )
    numerical_passed = (
        max_layer_relative_rmse
        <= relative_rmse_tolerance
        and min_layer_token_cosine
        >= min_token_cosine_tolerance
        and logits_metrics["relative_rmse"]
        <= relative_rmse_tolerance
        and continuous_next_token == cached_next_token
    )
    reconstructed_ids = (
        tokens.history_ids + tokens.continuation_ids
    )
    continuous_positions = list(
        range(
            start_position,
            start_position + len(tokens.full_ids),
        )
    )
    cached_positions = list(
        range(
            start_position,
            start_position + len(tokens.history_ids),
        )
    ) + list(
        range(
            start_position + len(tokens.history_ids),
            start_position + len(tokens.full_ids),
        )
    )
    attention_masks = {
        str(layer_index): _mask_metrics(
            torch,
            continuous_mask_capture.masks[layer_index],
            cached_mask_capture.masks[layer_index],
            history_tokens=len(tokens.history_ids),
        )
        for layer_index in mask_layer_indexes
    }
    token_sequence_exact = (
        reconstructed_ids == tokens.full_ids
    )
    position_sequence_exact = (
        cached_positions == continuous_positions
    )
    masks_exact = all(
        metrics.get("equal", False)
        for metrics in attention_masks.values()
    )
    continuation_starts_with_bos = (
        bool(tokens.continuation_ids)
        and tokens.continuation_ids[0]
        == getattr(
            getattr(model, "config", None),
            "bos_token_id",
            None,
        )
    )
    structural_passed = (
        token_sequence_exact
        and position_sequence_exact
        and masks_exact
        and not continuation_starts_with_bos
    )
    report = {
        "passed": numerical_passed,
        "numerical_passed": numerical_passed,
        "structural_passed": structural_passed,
        "start_position": start_position,
        "history_tokens": len(tokens.history_ids),
        "continuation_tokens": len(
            tokens.continuation_ids
        ),
        "full_tokens": len(tokens.full_ids),
        "continuation_first_token_id": (
            tokens.continuation_ids[0]
        ),
        "relative_rmse_tolerance": (
            relative_rmse_tolerance
        ),
        "min_token_cosine_tolerance": (
            min_token_cosine_tolerance
        ),
        "max_layer_relative_rmse": (
            max_layer_relative_rmse
        ),
        "min_layer_token_cosine": (
            min_layer_token_cosine
        ),
        "continuous_next_token_id": (
            continuous_next_token
        ),
        "cached_next_token_id": cached_next_token,
        "next_token_equal": (
            continuous_next_token
            == cached_next_token
        ),
        "logits": logits_metrics,
        "token_sequence": {
            "exact": token_sequence_exact,
            "first_mismatch": _first_mismatch(
                tokens.full_ids,
                reconstructed_ids,
            ),
            "continuous_ids": tokens.full_ids,
            "cache_split_history_ids": tokens.history_ids,
            "cache_split_continuation_ids": (
                tokens.continuation_ids
            ),
            "cache_split_reconstructed_ids": (
                reconstructed_ids
            ),
        },
        "position_sequence": {
            "exact": position_sequence_exact,
            "first_mismatch": _first_mismatch(
                continuous_positions,
                cached_positions,
            ),
            "continuous_positions": continuous_positions,
            "cache_split_positions": cached_positions,
        },
        "continuation_starts_with_bos": (
            continuation_starts_with_bos
        ),
        "attention_masks": attention_masks,
        "history_cache": history_cache_report,
        "first_layer_stages": _stage_diagnostics(
            torch,
            continuous_stage_capture.outputs,
            history_stage_capture.outputs,
            cached_stage_capture.outputs,
            history_tokens=history_tokens,
        ),
        "shape_kernel_probe": q_projection_shape_probe,
        "boundary_diagnostics": boundary_diagnostics,
        "layers": layers,
    }
    if return_continuous_output:
        return report, cached_output, continuous_output
    return report, cached_output
