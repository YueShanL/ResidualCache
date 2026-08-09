from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Sequence


_ARTICLE_HEADING = re.compile(r"^= ([^=].*?) =$", re.DOTALL)


@dataclass(frozen=True)
class WikiTextArticle:
    article_index: int
    title: str
    text: str


def iter_wikitext_articles(text_rows: Iterable[str]) -> Iterator[WikiTextArticle]:
    """Group WikiText raw rows at level-one headings without crossing articles."""

    article_index = -1
    title: str | None = None
    rows: list[str] = []
    for raw_text in text_rows:
        text = str(raw_text)
        match = _ARTICLE_HEADING.fullmatch(text.strip())
        if match is not None:
            if title is not None:
                yield WikiTextArticle(article_index, title, "".join(rows))
            article_index += 1
            title = match.group(1).strip()
            rows = [text]
        elif title is not None:
            rows.append(text)
    if title is not None:
        yield WikiTextArticle(article_index, title, "".join(rows))


def build_wikitext_sequences(
    articles: Iterable[WikiTextArticle],
    tokenizer: Any,
    *,
    sequence_length: int,
    sequence_count: int,
    article_stride: int = 1,
    seed: int = 13,
    split: str = "train",
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
) -> list[dict[str, Any]]:
    """Create exact-length, one-sequence-per-article token records."""

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if article_stride <= 0:
        raise ValueError("article_stride must be positive")

    selected: list[dict[str, Any]] = []
    stride_offset = seed % article_stride
    for article in articles:
        if article.article_index % article_stride != stride_offset:
            continue
        token_ids = list(tokenizer(article.text, add_special_tokens=True).input_ids)
        if len(token_ids) < sequence_length:
            continue
        selected.append(
            {
                "sequence_id": (
                    f"wikitext103-{split}-article-{article.article_index:07d}"
                ),
                "token_ids": [int(token_id) for token_id in token_ids[:sequence_length]],
                "source": dataset_name,
                "subset": dataset_config,
                "split": split,
                "article_index": article.article_index,
                "article_title": article.title,
                "token_start": 0,
                "token_end": sequence_length,
                "article_token_count": len(token_ids),
            }
        )
        if len(selected) == sequence_count:
            break
    if len(selected) != sequence_count:
        raise ValueError(
            f"found only {len(selected)} eligible articles; requested {sequence_count} "
            f"sequences of {sequence_length} tokens"
        )
    return selected


def _iter_arrow_texts(paths: Sequence[Path]) -> Iterator[str]:
    import pyarrow as pa

    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_stream(source)
            text_index = reader.schema.get_field_index("text")
            if text_index < 0:
                raise ValueError(f"Arrow dataset has no 'text' column: {path}")
            for batch in reader:
                for text in batch.column(text_index).to_pylist():
                    yield str(text)


def _iter_hf_texts(dataset: Any, *, batch_size: int = 10_000) -> Iterator[str]:
    if "text" not in dataset.column_names:
        raise ValueError("Hugging Face dataset has no 'text' column")
    for batch in dataset.iter(batch_size=batch_size):
        for text in batch["text"]:
            yield str(text)


def prepare_wikitext_jsonl(
    *,
    tokenizer_name: str,
    output_path: Path,
    split: str,
    sequence_length: int,
    sequence_count: int,
    article_stride: int,
    seed: int,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    arrow_directory: Path | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    manifest_path = output_path.with_suffix(".manifest.json")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    source_manifest: dict[str, Any]
    if arrow_directory is not None:
        arrow_paths = sorted(arrow_directory.glob(f"wikitext-{split}*.arrow"))
        if not arrow_paths:
            raise FileNotFoundError(
                f"no wikitext-{split}*.arrow files under {arrow_directory}"
            )
        text_rows = _iter_arrow_texts(arrow_paths)
        source_manifest = {
            "kind": "local_arrow",
            "arrow_files": [
                {"path": str(path.resolve()), "size_bytes": path.stat().st_size}
                for path in arrow_paths
            ],
        }
    else:
        from datasets import DownloadConfig, load_dataset

        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            download_config=DownloadConfig(local_files_only=local_files_only),
        )
        text_rows = _iter_hf_texts(dataset)
        source_manifest = {
            "kind": "huggingface",
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": local_files_only,
        }
    records = build_wikitext_sequences(
        iter_wikitext_articles(text_rows),
        tokenizer,
        sequence_length=sequence_length,
        sequence_count=sequence_count,
        article_stride=article_stride,
        seed=seed,
        split=split,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")

    manifest = {
        "schema_version": 1,
        "source": dataset_name,
        "subset": dataset_config,
        "split": split,
        "article_boundary_policy": "level-one WikiText headings; never cross articles",
        "chunk_policy": "first exact-length token chunk; at most one sequence per article",
        "sequence_count": len(records),
        "sequence_length": sequence_length,
        "article_stride": article_stride,
        "seed": seed,
        "tokenizer": {
            "requested_name": tokenizer_name,
            "resolved_name_or_path": tokenizer.name_or_path,
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "dataset_source": source_manifest,
        "output_jsonl": str(output_path.resolve()),
        "sequence_ids": [record["sequence_id"] for record in records],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare document-aligned WikiText raw sequences for learnable_index"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-name", default="Salesforce/wikitext")
    source.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--article-stride", type=int, default=37)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    manifest = prepare_wikitext_jsonl(
        arrow_directory=arguments.arrow_dir,
        tokenizer_name=arguments.tokenizer,
        output_path=arguments.output,
        split=arguments.split,
        sequence_length=arguments.sequence_length,
        sequence_count=arguments.sequences,
        article_stride=arguments.article_stride,
        seed=arguments.seed,
        dataset_name=arguments.dataset_name or "Salesforce/wikitext",
        dataset_config=arguments.dataset_config,
        cache_dir=arguments.cache_dir,
        local_files_only=not arguments.allow_network,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
