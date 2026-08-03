"""Standalone CAMELoT-style sequential perplexity evaluation for Gemma 4.

The runner follows the long-context language-model protocol used by CAMELoT:
the corpus is batchified into independent sequential streams, windows are
processed in order without a native Transformers KV cache, and external memory
is the only state carried between windows.  The reported task metric is
token-level perplexity.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterable, Iterator

from .gemma4_memory_adapter import (
    Gemma4MemoryAdapter,
    Gemma4MemoryAdapterConfig,
    parse_augmented_layers,
)
from .torch_token_memory import TokenMemoryConfig


LOCAL_GEMMA4_BASE = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "hf_cache"
    / "hub"
    / "models--google--gemma-4-E4B"
    / "snapshots"
    / "411aa17b749aa952df1359d2dcea73917a544d9a"
)

PAPER_DATASETS = {
    "wikitext103": {
        "name": "Salesforce/wikitext",
        "config": "wikitext-103-v1",
        "split": "test",
        "text_field": "text",
        "window_sizes": (256, 512, 1024),
    },
    "wikitext103_raw": {
        "name": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "test",
        "text_field": "text",
        "window_sizes": (256, 512, 1024),
    },
    "pg19": {
        "name": "deepmind/pg19",
        "config": None,
        "split": "test",
        "text_field": "text",
        "window_sizes": (512, 1024, 2048),
    },
    "pile_arxiv": {
        "name": "monology/pile-uncopyrighted",
        "config": None,
        "split": "test",
        "text_field": "text",
        "pile_subset": "ArXiv",
        "window_sizes": (512, 1024, 2048),
    },
}


@dataclass(frozen=True)
class EvaluationConfig:
    model_name: str
    methods: tuple[str, ...] = ("base", "camelot", "vmf_records")
    window_sizes: tuple[int, ...] = (512, 1024, 2048)
    batch_size: int = 4
    max_tokens: int | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    memory_device: str | None = None
    seed: int = 1234
    add_bos: bool = True
    position_policy: str = "cache_relative"
    show_progress: bool = True

    def __post_init__(self) -> None:
        allowed = {"base", "camelot", "vmf_records"}
        unknown = set(self.methods) - allowed
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        if not self.methods:
            raise ValueError("At least one method is required.")
        if not self.window_sizes or min(self.window_sizes) <= 1:
            raise ValueError("Window sizes must all be greater than one.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.max_tokens is not None and self.max_tokens <= self.batch_size:
            raise ValueError("max_tokens must exceed batch_size.")
        if self.position_policy not in {
            "cache_relative",
            "continuous",
            "window_reset",
        }:
            raise ValueError(f"Unknown position_policy {self.position_policy!r}.")


def _require_model_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised by the CLI.
        raise RuntimeError(
            "Evaluation requires torch and transformers from the project environment."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _torch_dtype(torch, name: str):
    if name == "auto":
        return "auto"
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype {name!r}.") from exc


class _NullProgressBar:
    """Small tqdm-compatible fallback used when progress is disabled."""

    def update(self, _amount: int = 1) -> None:
        return None

    def set_postfix(self, **_values) -> None:
        return None

    def close(self) -> None:
        return None


def _progress_bar(
    *,
    enabled: bool,
    total: int | None,
    description: str,
    unit: str,
    leave: bool = True,
    position: int = 0,
):
    if not enabled:
        return _NullProgressBar()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NullProgressBar()
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        position=position,
    )


def load_local_gemma4(config: EvaluationConfig):
    torch, AutoModelForCausalLM, AutoTokenizer = _require_model_dependencies()
    model_path = Path(config.model_name).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local Gemma 4 snapshot does not exist: {model_path}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {
        "dtype": _torch_dtype(torch, config.dtype),
        "local_files_only": True,
        "attn_implementation": "eager",
    }
    if config.device == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    if config.device != "auto":
        model.to(config.device)
    model.eval()
    return torch, tokenizer, model


def _row_matches_pile_subset(row: dict, subset: str) -> bool:
    metadata = row.get("meta") or row.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return subset.lower() in metadata.lower()
    name = (
        metadata.get("pile_set_name")
        or metadata.get("subset")
        or metadata.get("source")
        or ""
    )
    return str(name).lower() == subset.lower()


def _iter_local_rows(
    path: Path, *, text_field: str, split: str
) -> Iterator[dict]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                payload = json.load(handle)
                rows = payload if isinstance(payload, list) else payload[split]
                yield from rows
            else:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        return
    if path.suffix.lower() in {".txt", ".text"}:
        yield {text_field: path.read_text(encoding="utf-8")}
        return
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:  # pragma: no cover - exercised by the CLI.
        raise RuntimeError("Parquet or saved datasets require the datasets package.") from exc
    if path.is_dir():
        dataset = load_from_disk(str(path))
        if hasattr(dataset, "keys"):
            dataset = dataset[split]
    elif path.suffix.lower() == ".parquet":
        dataset = load_dataset(
            "parquet", data_files={split: str(path)}, split=split
        )
    else:
        raise ValueError(f"Unsupported local dataset path: {path}")
    yield from dataset


def iter_dataset_texts(
    *,
    preset: str,
    dataset_path: Path | None = None,
    split: str | None = None,
    text_field: str | None = None,
    max_documents: int | None = None,
    streaming: bool = False,
) -> Iterator[str]:
    if preset not in PAPER_DATASETS:
        raise ValueError(
            f"Unknown dataset preset {preset!r}; choose from {sorted(PAPER_DATASETS)}."
        )
    specification = PAPER_DATASETS[preset]
    selected_split = split or specification["split"]
    selected_field = text_field or specification["text_field"]
    if dataset_path is not None:
        rows: Iterable[dict] = _iter_local_rows(
            dataset_path, text_field=selected_field, split=selected_split
        )
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - exercised by the CLI.
            raise RuntimeError("Hugging Face dataset presets require datasets.") from exc
        arguments = [specification["name"]]
        if specification["config"] is not None:
            arguments.append(specification["config"])
        rows = load_dataset(
            *arguments,
            split=selected_split,
            streaming=streaming,
        )
    emitted = 0
    pile_subset = specification.get("pile_subset")
    for row in rows:
        if pile_subset and not _row_matches_pile_subset(row, pile_subset):
            continue
        text = row.get(selected_field)
        if text is None:
            raise KeyError(
                f"Dataset row has no text field {selected_field!r}: {sorted(row)}"
            )
        if text:
            yield str(text)
            emitted += 1
            if max_documents is not None and emitted >= max_documents:
                return


def tokenize_corpus(
    tokenizer,
    texts: Iterable[str],
    *,
    max_tokens: int | None,
    add_bos: bool = True,
    show_progress: bool = True,
):
    """Tokenize a corpus once so every evaluated method sees identical tokens."""

    torch, _AutoModel, _AutoTokenizer = _require_model_dependencies()
    token_ids: list[int] = (
        [int(tokenizer.bos_token_id)]
        if add_bos and tokenizer.bos_token_id is not None
        else []
    )
    separator = (
        [int(tokenizer.eos_token_id)]
        if tokenizer.eos_token_id is not None
        else []
    )
    progress = _progress_bar(
        enabled=show_progress,
        total=None,
        description="[2/5] Read and tokenize corpus",
        unit="doc",
    )
    try:
        for text in texts:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            ).input_ids
            progress.update()
            if not encoded:
                continue
            if token_ids and separator:
                token_ids.extend(separator)
            token_ids.extend(int(token_id) for token_id in encoded)
            progress.set_postfix(tokens=f"{len(token_ids):,}")
            if max_tokens is not None and len(token_ids) >= max_tokens:
                del token_ids[max_tokens:]
                break
    finally:
        progress.close()
    if len(token_ids) < 2:
        raise ValueError("The selected corpus produced fewer than two tokens.")
    return torch.tensor(token_ids, dtype=torch.long)


def contiguous_stream_batchify(tokens, batch_size: int):
    """Split one token stream into independent contiguous batch streams."""

    usable = (tokens.numel() // batch_size) * batch_size
    if usable < batch_size * 2:
        raise ValueError("The corpus is too short for the requested batch size.")
    return tokens[:usable].view(batch_size, -1).contiguous()


def sequential_windows(streams, window_size: int):
    """Yield disjoint inputs and their one-token-ahead targets."""

    stream_length = streams.shape[1]
    position = 0
    window_index = 0
    while position < stream_length - 1:
        length = min(window_size, stream_length - 1 - position)
        if length <= 0:
            break
        yield (
            window_index,
            position,
            streams[:, position : position + length],
            streams[:, position + 1 : position + length + 1],
        )
        position += length
        window_index += 1


def _model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def window_position_offset(
    *,
    method: str,
    policy: str,
    window_index: int,
    stream_position: int,
    window_size: int,
) -> int:
    if method == "base" or policy == "window_reset" or window_index == 0:
        return 0
    if policy == "cache_relative":
        return window_size
    if policy == "continuous":
        return stream_position
    raise ValueError(f"Unknown position policy {policy!r}.")


def evaluate_streams(
    *,
    torch,
    model,
    streams,
    method: str,
    window_size: int,
    memory_config: TokenMemoryConfig | None,
    adapter_config: Gemma4MemoryAdapterConfig | None,
    position_policy: str,
    show_progress: bool,
) -> tuple[dict, list[dict]]:
    if method == "base":
        context = nullcontext(None)
    else:
        if memory_config is None:
            raise ValueError("A memory method requires memory_config.")
        context = Gemma4MemoryAdapter(model, memory_config, adapter_config)
    input_device = _model_input_device(model)
    if input_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(input_device)
    total_nll = 0.0
    total_tokens = 0
    window_rows: list[dict] = []
    started = time.perf_counter()
    window_count = math.ceil((streams.shape[1] - 1) / window_size)
    progress = _progress_bar(
        enabled=show_progress,
        total=window_count,
        description=f"[4/5] {method} - L={window_size}",
        unit="window",
        leave=False,
        position=1,
    )
    try:
        with context as controller, torch.inference_mode():
            for window_index, position, input_ids, targets in sequential_windows(
                streams, window_size
            ):
                input_ids = input_ids.to(input_device)
                targets = targets.to(input_device)
                position_offset = window_position_offset(
                    method=method,
                    policy=position_policy,
                    window_index=window_index,
                    stream_position=position,
                    window_size=window_size,
                )
                position_ids = torch.arange(
                    position_offset,
                    position_offset + input_ids.shape[1],
                    device=input_device,
                    dtype=torch.long,
                ).unsqueeze(0)
                before = time.perf_counter()
                output = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                logits = output.logits
                loss_sum = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )
                if input_device.type == "cuda":
                    torch.cuda.synchronize(input_device)
                elapsed = time.perf_counter() - before
                token_count = int(targets.numel())
                nll = float(loss_sum.item())
                total_nll += nll
                total_tokens += token_count
                window_rows.append(
                    {
                        "method": method,
                        "window_size": window_size,
                        "window_index": window_index,
                        "stream_position": position,
                        "position_offset": position_offset,
                        "tokens": token_count,
                        "negative_log_likelihood": nll,
                        "perplexity": math.exp(min(700.0, nll / token_count)),
                        "seconds": elapsed,
                    }
                )
                progress.update()
                progress.set_postfix(
                    ppl=f"{math.exp(min(700.0, total_nll / total_tokens)):.3f}",
                    tokens=f"{total_tokens:,}",
                )
            controller_snapshot = (
                controller.snapshot() if controller is not None else None
            )
    finally:
        progress.close()
    total_seconds = time.perf_counter() - started
    mean_nll = total_nll / total_tokens
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(input_device))
        if input_device.type == "cuda"
        else None
    )
    summary = {
        "method": method,
        "window_size": window_size,
        "batch_size": int(streams.shape[0]),
        "stream_tokens": int(streams.numel()),
        "scored_tokens": total_tokens,
        "negative_log_likelihood": total_nll,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": math.exp(min(700.0, mean_nll)),
        "seconds": total_seconds,
        "tokens_per_second": total_tokens / total_seconds,
        "peak_model_device_bytes": peak_bytes,
        "memory": controller_snapshot,
    }
    return summary, window_rows


def validate_first_window_equivalence(
    window_rows: list[dict], *, relative_tolerance: float = 1e-6
) -> dict:
    """Fail if an empty-memory method differs from base on its first window."""

    first_rows = [row for row in window_rows if row["window_index"] == 0]
    base_by_window = {
        row["window_size"]: row["negative_log_likelihood"]
        for row in first_rows
        if row["method"] == "base"
    }
    if not base_by_window:
        return {
            "checked": False,
            "reason": "base was not selected",
            "maximum_absolute_nll_difference": None,
        }
    maximum_difference = 0.0
    checked_methods = 0
    for row in first_rows:
        if row["method"] == "base":
            continue
        baseline = base_by_window.get(row["window_size"])
        if baseline is None:
            continue
        difference = abs(row["negative_log_likelihood"] - baseline)
        maximum_difference = max(maximum_difference, difference)
        tolerance = relative_tolerance * max(1.0, abs(baseline))
        if difference > tolerance:
            raise RuntimeError(
                "First-window equivalence failed for "
                f"{row['method']} at L={row['window_size']}: "
                f"|NLL - base|={difference} > {tolerance}. This usually means "
                "the memory was read after an early write or causality was lost."
            )
        checked_methods += 1
    return {
        "checked": True,
        "checked_method_window_pairs": checked_methods,
        "relative_tolerance": relative_tolerance,
        "maximum_absolute_nll_difference": maximum_difference,
    }


def run_evaluation(
    *,
    config: EvaluationConfig,
    texts: Iterable[str],
    output_dir: Path,
    memory_template: TokenMemoryConfig,
    adapter_config: Gemma4MemoryAdapterConfig,
) -> dict:
    random.seed(config.seed)
    model_progress = _progress_bar(
        enabled=config.show_progress,
        total=1,
        description="[1/5] Load local Gemma 4",
        unit="model",
    )
    try:
        torch, tokenizer, model = load_local_gemma4(config)
        model_progress.update()
    finally:
        model_progress.close()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    tokens = tokenize_corpus(
        tokenizer,
        texts,
        max_tokens=config.max_tokens,
        add_bos=config.add_bos,
        show_progress=config.show_progress,
    )
    batchify_progress = _progress_bar(
        enabled=config.show_progress,
        total=1,
        description="[3/5] Batchify contiguous streams",
        unit="corpus",
    )
    try:
        streams = contiguous_stream_batchify(tokens, config.batch_size)
        batchify_progress.set_postfix(
            batch=config.batch_size,
            tokens=f"{streams.numel():,}",
        )
        batchify_progress.update()
    finally:
        batchify_progress.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    window_rows: list[dict] = []
    experiment_progress = _progress_bar(
        enabled=config.show_progress,
        total=len(config.window_sizes) * len(config.methods),
        description="[4/5] Evaluation matrix",
        unit="run",
        position=0,
    )
    try:
        for window_size in config.window_sizes:
            for method in config.methods:
                selected_memory = (
                    None
                    if method == "base"
                    else TokenMemoryConfig(
                        **{
                            **asdict(memory_template),
                            "method": method,
                        }
                    )
                )
                summary, rows = evaluate_streams(
                    torch=torch,
                    model=model,
                    streams=streams,
                    method=method,
                    window_size=window_size,
                    memory_config=selected_memory,
                    adapter_config=adapter_config,
                    position_policy=config.position_policy,
                    show_progress=config.show_progress,
                )
                summaries.append(summary)
                window_rows.extend(rows)
                experiment_progress.update()
                experiment_progress.set_postfix(
                    last=f"{method}/L={window_size}",
                    ppl=f"{summary['perplexity']:.3f}",
                )
    finally:
        experiment_progress.close()
    baselines = {
        row["window_size"]: row["perplexity"]
        for row in summaries
        if row["method"] == "base"
    }
    for row in summaries:
        baseline = baselines.get(row["window_size"])
        row["perplexity_change_vs_base"] = (
            row["perplexity"] - baseline if baseline is not None else None
        )
        row["relative_perplexity_change_vs_base"] = (
            row["perplexity"] / baseline - 1.0
            if baseline is not None
            else None
        )
    first_window_check = validate_first_window_equivalence(window_rows)
    token_sha256 = hashlib.sha256(
        streams.contiguous().numpy().tobytes()
    ).hexdigest()
    manifest = {
        "protocol": {
            "name": "CAMELoT sequential windowed causal language modeling",
            "metric": "token-level perplexity",
            "native_kv_cache": False,
            "memory_order": "retrieve, attend, then write current window",
            "vmf_write_commit": {
                "chunk_size": memory_template.vmf_write_chunk_size,
                "definition": (
                    "tokens in each post-attention chunk compute posterior "
                    "assignments against the same pre-commit memory state; "
                    "chunks are committed sequentially"
                ),
                "exact_token_sequential_setting": 1,
            },
            "batchification": "Transformer-XL contiguous independent streams",
            "position_ids": {
                "policy": config.position_policy,
                "cache_relative_definition": (
                    "base and first memory window use 0..L-1; later memory "
                    "windows use L..2L-1, matching a retrieved length-L past cache"
                ),
            },
            "paper_reference": "https://arxiv.org/abs/2402.13449",
            "paper_default_threshold": 0.93,
            "paper_default_slots_per_layer": 10_000,
            "paper_default_batch_size": 4,
            "paper_main_window_sizes": {
                "wikitext103": [256, 512, 1024],
                "pg19_and_pile_arxiv": [512, 1024, 2048],
            },
            "bos_policy": (
                "one BOS token at the beginning of the batchified corpus"
                if config.add_bos
                else "no BOS token inserted"
            ),
            "model_deviation": "local Gemma 4 E4B replaces the paper's LLaMA2-7B",
        },
        "evaluation_config": asdict(config),
        "memory_template": asdict(memory_template),
        "adapter_config": asdict(adapter_config),
        "token_count_before_batchify": int(tokens.numel()),
        "stream_shape": list(streams.shape),
        "stream_token_sha256": token_sha256,
        "first_window_equivalence": first_window_check,
        "results": summaries,
    }
    reporting_progress = _progress_bar(
        enabled=config.show_progress,
        total=1,
        description="[5/5] Write validated results",
        unit="report",
    )
    try:
        (output_dir / "summary.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (output_dir / "windows.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for row in window_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        reporting_progress.set_postfix(path=str(output_dir))
        reporting_progress.update()
    finally:
        reporting_progress.close()
    return manifest


def _csv_strings(text: str) -> tuple[str, ...]:
    return tuple(piece.strip() for piece in text.split(",") if piece.strip())


def _csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in text.split(",") if piece.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate token memories inside local Gemma 4 using CAMELoT's CLM protocol."
    )
    parser.add_argument("--model-name", default=str(LOCAL_GEMMA4_BASE))
    parser.add_argument(
        "--methods",
        default="base,camelot,vmf_records",
    )
    parser.add_argument(
        "--window-sizes",
        help=(
            "Comma-separated override. Defaults to 256/512/1024 for WikiText "
            "and 512/1024/2048 for PG-19 or Pile-ArXiv."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-device")
    parser.add_argument("--seed", type=int, default=1234)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="show_progress",
        action="store_true",
        default=True,
        help="Show stage and per-window progress bars (default).",
    )
    progress_group.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable all progress bars.",
    )
    parser.add_argument("--no-bos", action="store_true")
    parser.add_argument(
        "--position-policy",
        choices=("cache_relative", "continuous", "window_reset"),
        default="cache_relative",
        help=(
            "RoPE position policy. cache_relative matches the paper's use of "
            "retrieved K/V as a length-L past cache."
        ),
    )
    parser.add_argument(
        "--dataset-preset",
        choices=sorted(PAPER_DATASETS),
        default="wikitext103",
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--text-field")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slot-capacity", type=int, default=10_000)
    parser.add_argument("--record-capacity", type=int, default=10_000)
    parser.add_argument("--camelot-threshold", type=float, default=0.93)
    parser.add_argument("--route-top-k", type=int, default=4)
    parser.add_argument(
        "--vmf-write-chunk-size",
        type=int,
        default=32,
        help=(
            "Number of post-attention writes committed per vMF posterior "
            "mini-batch. Use 1 for exact token-sequential routing."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tau-new", type=float, default=0.5)
    parser.add_argument("--count-exponent", type=float, default=0.5)
    parser.add_argument("--concentration-prior-mass", type=float, default=1.0)
    parser.add_argument("--maximum-concentration", type=float, default=1_000.0)
    parser.add_argument(
        "--augmented-layers",
        help="Comma-separated indices/ranges, for example 0-5,11; default: all.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.hf_home is not None:
        os.environ["HF_HOME"] = str(arguments.hf_home.resolve())
    window_sizes = (
        _csv_ints(arguments.window_sizes)
        if arguments.window_sizes
        else tuple(PAPER_DATASETS[arguments.dataset_preset]["window_sizes"])
    )
    config = EvaluationConfig(
        model_name=arguments.model_name,
        methods=_csv_strings(arguments.methods),
        window_sizes=window_sizes,
        batch_size=arguments.batch_size,
        max_tokens=arguments.max_tokens,
        dtype=arguments.dtype,
        device=arguments.device,
        memory_device=arguments.memory_device,
        seed=arguments.seed,
        add_bos=not arguments.no_bos,
        position_policy=arguments.position_policy,
        show_progress=arguments.show_progress,
    )
    memory_template = TokenMemoryConfig(
        method="camelot",
        slot_capacity=arguments.slot_capacity,
        record_capacity=arguments.record_capacity,
        camelot_threshold=arguments.camelot_threshold,
        route_top_k=arguments.route_top_k,
        vmf_write_chunk_size=arguments.vmf_write_chunk_size,
        alpha=arguments.alpha,
        tau_new=arguments.tau_new,
        count_exponent=arguments.count_exponent,
        concentration_prior_mass=arguments.concentration_prior_mass,
        maximum_concentration=arguments.maximum_concentration,
    )
    adapter_config = Gemma4MemoryAdapterConfig(
        memory_device=arguments.memory_device,
        augmented_layers=parse_augmented_layers(arguments.augmented_layers),
    )
    texts = iter_dataset_texts(
        preset=arguments.dataset_preset,
        dataset_path=arguments.dataset_path,
        split=arguments.split,
        text_field=arguments.text_field,
        max_documents=arguments.max_documents,
        streaming=arguments.streaming,
    )
    manifest = run_evaluation(
        config=config,
        texts=texts,
        output_dir=arguments.output_dir,
        memory_template=memory_template,
        adapter_config=adapter_config,
    )
    compact_results = [
        {
            "method": row["method"],
            "window_size": row["window_size"],
            "perplexity": row["perplexity"],
            "relative_perplexity_change_vs_base": row[
                "relative_perplexity_change_vs_base"
            ],
            "tokens_per_second": row["tokens_per_second"],
            "external_memory_bytes": (
                row["memory"]["memory_bytes"] if row["memory"] else 0
            ),
        }
        for row in manifest["results"]
    ]
    print(json.dumps(compact_results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
