from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


SHIFT_SUITES = {"lufy_shift", "convomem_repeated_question"}
SAME_QUESTION_DIFFERENT_FACT_SUITES = {"convomem_repeated_question"}
QUERY_TO_FACT_SUITES = {"convomem_query_to_fact"}


@dataclass(frozen=True)
class AnalysisConfig:
    collect_dir: Path
    output_dir: Path | None = None
    top_k: int = 10


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cosine_matrix(torch, vectors):
    normalized = torch.nn.functional.normalize(vectors.float(), dim=-1)
    return normalized @ normalized.T


def _centered_cosine_matrix(torch, vectors):
    centered = vectors.float() - vectors.float().mean(dim=-1, keepdim=True)
    normalized = torch.nn.functional.normalize(centered, dim=-1)
    return normalized @ normalized.T


def _normalized_euclidean_matrix(torch, vectors):
    vectors = vectors.float()
    normalized = (vectors - vectors.mean(dim=0, keepdim=True)) / vectors.std(dim=0, keepdim=True).clamp_min(1e-6)
    return torch.cdist(normalized, normalized)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _answer_mean_state(torch, state, tensor_row: dict):
    indices = tensor_row["answer_indices"]
    if not indices:
        return None
    return state[:, indices, :].mean(dim=1)


def _load_vectors(torch, collect_dir: Path, metadata: list[dict]):
    loaded = []
    for row in metadata:
        tensor_row = torch.load(collect_dir / row["tensor_path"], map_location="cpu")
        states = tensor_row["states"]
        anchor_position_family = tensor_row.get("anchor_position_family", "final_prompt")
        for target, state in states.items():
            loaded.append((row, target, anchor_position_family, state[:, tensor_row["final_prompt_index"], :]))
            answer_state = _answer_mean_state(torch, state, tensor_row)
            if answer_state is not None:
                loaded.append((row, target, "answer", answer_state))
    return loaded


def _group_loaded(loaded):
    grouped = defaultdict(list)
    for row, target, position_family, vectors in loaded:
        grouped[(target, position_family)].append((row, vectors))
    return grouped


def _same_different_report(torch, rows_and_vectors: list[tuple[dict, object]]) -> list[dict]:
    if not rows_and_vectors:
        return []
    layer_count = rows_and_vectors[0][1].shape[0]
    report = []
    for layer in range(layer_count):
        vectors = torch.stack([vectors[layer] for _row, vectors in rows_and_vectors])
        sims = _cosine_matrix(torch, vectors)
        centered_sims = _centered_cosine_matrix(torch, vectors)
        distances = _normalized_euclidean_matrix(torch, vectors)
        buckets = defaultdict(list)
        centered_buckets = defaultdict(list)
        distance_buckets = defaultdict(list)
        for i, (left, _lv) in enumerate(rows_and_vectors):
            for j, (right, _rv) in enumerate(rows_and_vectors):
                if i >= j:
                    continue
                target_bucket = "same_target" if left["target_fact_id"] == right["target_fact_id"] else "different_target"
                buckets[target_bucket].append(float(sims[i, j]))
                centered_buckets[target_bucket].append(float(centered_sims[i, j]))
                distance_buckets[target_bucket].append(float(distances[i, j]))
                buckets["same_task" if left["task_id"] == right["task_id"] else "different_task"].append(float(sims[i, j]))
                buckets["same_mode" if left["mode"] == right["mode"] else "different_mode"].append(float(sims[i, j]))
                buckets["conflict_pair" if left["conflict"] or right["conflict"] else "non_conflict_pair"].append(float(sims[i, j]))
        correct_vectors = [vectors[index] for index, (row, _v) in enumerate(rows_and_vectors) if row.get("correct")]
        wrong_vectors = [vectors[index] for index, (row, _v) in enumerate(rows_and_vectors) if not row.get("correct")]
        correct_wrong_cosine = []
        if correct_vectors and wrong_vectors:
            cw = _cosine_matrix(torch, torch.stack(correct_vectors + wrong_vectors))
            split = len(correct_vectors)
            correct_wrong_cosine = cw[:split, split:].flatten().tolist()
        report.append(
            {
                "layer": layer,
                **{f"mean_{name}_cosine": _mean(values) for name, values in buckets.items()},
                **{f"mean_{name}_centered_cosine": _mean(values) for name, values in centered_buckets.items()},
                **{f"mean_{name}_normalized_euclidean": _mean(values) for name, values in distance_buckets.items()},
                "same_target_margin": _mean(buckets["same_target"]) - _mean(buckets["different_target"]),
                "same_target_centered_margin": _mean(centered_buckets["same_target"]) - _mean(centered_buckets["different_target"]),
                "same_target_distance_margin": _mean(distance_buckets["different_target"]) - _mean(distance_buckets["same_target"]),
                "same_task_margin": _mean(buckets["same_task"]) - _mean(buckets["different_task"]),
                "mean_correct_wrong_cosine": _mean([float(value) for value in correct_wrong_cosine]),
            }
        )
    return report


def _retrieval_report(torch, rows_and_vectors: list[tuple[dict, object]], top_k: int) -> list[dict]:
    if len(rows_and_vectors) < 2:
        return []
    layer_count = rows_and_vectors[0][1].shape[0]
    report = []
    for layer in range(layer_count):
        vectors = torch.stack([vectors[layer] for _row, vectors in rows_and_vectors])
        sims = _cosine_matrix(torch, vectors)
        hits_1 = hits_k = false_conflict = conflict_queries = queries = 0
        confusion = Counter()
        for i, (query, _qv) in enumerate(rows_and_vectors):
            scores = [(float(sims[i, j]), j) for j in range(len(rows_and_vectors)) if i != j]
            scores.sort(reverse=True)
            ranked = [rows_and_vectors[j][0] for _score, j in scores]
            if not ranked:
                continue
            queries += 1
            nearest = ranked[0]
            hits_1 += int(nearest["target_fact_id"] == query["target_fact_id"])
            hits_k += int(any(row["target_fact_id"] == query["target_fact_id"] for row in ranked[:top_k]))
            if query["conflict"]:
                conflict_queries += 1
                if nearest["entity"] == query["entity"] and nearest["target_fact_id"] != query["target_fact_id"]:
                    false_conflict += 1
            confusion[(query["target_fact_id"], nearest["target_fact_id"])] += 1
        report.append(
            {
                "layer": layer,
                "queries": queries,
                "top1_same_target_accuracy": hits_1 / queries if queries else 0.0,
                f"top{top_k}_same_target_recall": hits_k / queries if queries else 0.0,
                "conflict_queries": conflict_queries,
                "conflict_false_positive_rate": false_conflict / conflict_queries if conflict_queries else 0.0,
                "confusion": [
                    {"query_target": left, "retrieved_target": right, "count": count}
                    for (left, right), count in confusion.most_common()
                ],
            }
        )
    return report


def _query_to_fact_report(torch, rows_and_vectors: list[tuple[dict, object]], top_k: int) -> list[dict]:
    refs = [(row, vectors) for row, vectors in rows_and_vectors if row.get("suite") in QUERY_TO_FACT_SUITES and row["condition_id"] == "fact_reference"]
    queries = [(row, vectors) for row, vectors in rows_and_vectors if row.get("suite") in QUERY_TO_FACT_SUITES and row["condition_id"] == "question_query"]
    if not refs or not queries:
        return []

    layer_count = rows_and_vectors[0][1].shape[0]
    report = []
    for layer in range(layer_count):
        ref_vectors = torch.stack([vectors[layer] for _row, vectors in refs])
        query_vectors = torch.stack([vectors[layer] for _row, vectors in queries])
        sims = _cosine_matrix(torch, torch.cat([query_vectors, ref_vectors], dim=0))[: len(queries), len(queries) :]
        hits_1 = hits_k = 0
        same_fact = []
        same_question_other_fact = []
        other_fact = []
        for query_index, (query, _qv) in enumerate(queries):
            scores = [(float(sims[query_index, ref_index]), ref_index) for ref_index in range(len(refs))]
            scores.sort(reverse=True)
            ranked = [refs[ref_index][0] for _score, ref_index in scores]
            hits_1 += int(ranked[0]["target_fact_id"] == query["target_fact_id"])
            hits_k += int(any(row["target_fact_id"] == query["target_fact_id"] for row in ranked[:top_k]))
            for score, ref_index in scores:
                ref = refs[ref_index][0]
                if ref["target_fact_id"] == query["target_fact_id"]:
                    same_fact.append(score)
                elif ref.get("question_key") == query.get("question_key"):
                    same_question_other_fact.append(score)
                else:
                    other_fact.append(score)
        query_count = len(queries)
        report.append(
            {
                "layer": layer,
                "queries": query_count,
                "fact_references": len(refs),
                "top1_query_to_fact_accuracy": hits_1 / query_count if query_count else 0.0,
                f"top{top_k}_query_to_fact_recall": hits_k / query_count if query_count else 0.0,
                "mean_query_same_fact_cosine": _mean(same_fact),
                "mean_query_same_question_other_fact_cosine": _mean(same_question_other_fact),
                "mean_query_other_fact_cosine": _mean(other_fact),
                "query_to_fact_margin": _mean(same_fact) - _mean(other_fact),
                "query_to_same_question_margin": _mean(same_fact) - _mean(same_question_other_fact),
            }
        )
    return report


def _suite_report(torch, rows_and_vectors: list[tuple[dict, object]]) -> list[dict]:
    if len(rows_and_vectors) < 2:
        return []
    layer_count = rows_and_vectors[0][1].shape[0]
    report = []
    for layer in range(layer_count):
        vectors = torch.stack([vectors[layer] for _row, vectors in rows_and_vectors])
        sims = _cosine_matrix(torch, vectors)
        shift = []
        same_question_different_fact = []
        for i, (left, _lv) in enumerate(rows_and_vectors):
            for j, (right, _rv) in enumerate(rows_and_vectors):
                if i >= j:
                    continue
                sim = float(sims[i, j])
                if (
                    left.get("suite") in SHIFT_SUITES
                    and right.get("suite") in SHIFT_SUITES
                    and left.get("suite") == right.get("suite")
                    and left["target_fact_id"] == right["target_fact_id"]
                    and left["condition_id"] != right["condition_id"]
                ):
                    shift.append(sim)
                if (
                    left.get("suite") in SAME_QUESTION_DIFFERENT_FACT_SUITES
                    and right.get("suite") in SAME_QUESTION_DIFFERENT_FACT_SUITES
                    and left.get("suite") == right.get("suite")
                    and left.get("question_key") == right.get("question_key")
                    and left["target_fact_id"] != right["target_fact_id"]
                    and left["condition_id"] == right["condition_id"]
                ):
                    same_question_different_fact.append(sim)
        if shift or same_question_different_fact:
            report.append(
                {
                    "layer": layer,
                    "shift_pairs": len(shift),
                    "shift_mean_cosine": _mean(shift),
                    "shift_mean_distance": 1.0 - _mean(shift) if shift else 0.0,
                    "same_question_different_fact_pairs": len(same_question_different_fact),
                    "same_question_different_fact_mean_cosine": _mean(same_question_different_fact),
                    "fact_separation_margin": _mean(shift) - _mean(same_question_different_fact),
                }
            )
    return report


def _recommend(best_recall: float, best_conflict_fp: float, canonical_margin: float, top_k: int) -> str:
    if best_recall >= 0.80 and best_conflict_fp <= 0.10:
        return "viable"
    if canonical_margin > 0 and best_recall >= 0.60:
        return "viable only with canonical protocol"
    if best_recall >= 0.40:
        return "viable only for procedural/task memory"
    return "not viable for factual recall"


def _plot_series(rows: list[dict], value_key: str, output_path: Path, title: str, ylabel: str) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for target in sorted({row["target"] for row in rows}):
        for position in sorted({row["position_family"] for row in rows if row["target"] == target}):
            series = sorted(
                [row for row in rows if row["target"] == target and row["position_family"] == position],
                key=lambda row: row["layer"],
            )
            values = [row.get(value_key) for row in series]
            if not any(value is not None for value in values):
                continue
            ax.plot([row["layer"] for row in series], values, marker="o", linewidth=1.4, markersize=3, label=f"{target}:{position}")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_outputs(summary_rows: list[dict], retrieval_rows: list[dict], output_dir: Path, top_k: int) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    _plot_series(
        summary_rows,
        "same_target_margin",
        plot_dir / "same_target_cosine_margin.png",
        "Same-target vs Different-target Cosine Margin",
        "Cosine margin",
    )
    _plot_series(
        summary_rows,
        "same_target_distance_margin",
        plot_dir / "same_target_distance_margin.png",
        "Different-target vs Same-target Normalized Distance Margin",
        "Distance margin",
    )
    _plot_series(
        retrieval_rows,
        f"top{top_k}_same_target_recall",
        plot_dir / f"top{top_k}_same_target_recall.png",
        f"Top-{top_k} Same-target Recall",
        "Recall",
    )
    _plot_series(
        retrieval_rows,
        "conflict_false_positive_rate",
        plot_dir / "conflict_false_positive_rate.png",
        "Conflict False-positive Rate",
        "False-positive rate",
    )


def _plot_suite_outputs(suite_rows: list[dict], output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    _plot_series(
        suite_rows,
        "shift_mean_distance",
        plot_dir / "residual_shift_distance.png",
        "Same-fact Residual Shift Across Context Conditions",
        "1 - cosine",
    )
    _plot_series(
        suite_rows,
        "fact_separation_margin",
        plot_dir / "same_question_fact_separation_margin.png",
        "Same-fact Shift vs Same-question Different-fact Separation",
        "Cosine margin",
    )


def _plot_query_to_fact_outputs(query_rows: list[dict], output_dir: Path, top_k: int) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    _plot_series(
        query_rows,
        f"top{top_k}_query_to_fact_recall",
        plot_dir / f"top{top_k}_query_to_fact_recall.png",
        f"Top-{top_k} Query-to-fact Recall",
        "Recall",
    )
    _plot_series(
        query_rows,
        "query_to_fact_margin",
        plot_dir / "query_to_fact_margin.png",
        "Query State vs Fact-reference State Margin",
        "Cosine margin",
    )


def analyze(config: AnalysisConfig) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Analysis requires torch from the current environment.") from exc

    output_dir = config.output_dir or (config.collect_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_jsonl(config.collect_dir / "metadata.jsonl")
    loaded = _load_vectors(torch, config.collect_dir, metadata)
    grouped = _group_loaded(loaded)

    summary_rows = []
    retrieval_rows = []
    suite_rows = []
    query_to_fact_rows = []
    for (target, position_family), rows_and_vectors in sorted(grouped.items()):
        for row in _same_different_report(torch, rows_and_vectors):
            summary_rows.append({"target": target, "position_family": position_family, **row})
        for row in _retrieval_report(torch, rows_and_vectors, config.top_k):
            retrieval_rows.append({"target": target, "position_family": position_family, **row})
        for row in _suite_report(torch, rows_and_vectors):
            suite_rows.append({"target": target, "position_family": position_family, **row})
        for row in _query_to_fact_report(torch, rows_and_vectors, config.top_k):
            query_to_fact_rows.append({"target": target, "position_family": position_family, **row})

    _write_jsonl(output_dir / "layer_similarity_report.jsonl", summary_rows)
    (output_dir / "retrieval_report.json").write_text(json.dumps(retrieval_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "suite_report.json").write_text(json.dumps(suite_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "query_to_fact_report.json").write_text(json.dumps(query_to_fact_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_outputs(summary_rows, retrieval_rows, output_dir, config.top_k)
    _plot_suite_outputs(suite_rows, output_dir)
    _plot_query_to_fact_outputs(query_to_fact_rows, output_dir, config.top_k)

    query_best = max(query_to_fact_rows, key=lambda row: row[f"top{config.top_k}_query_to_fact_recall"], default={})
    best = max(retrieval_rows, key=lambda row: row[f"top{config.top_k}_same_target_recall"], default={})
    canonical_margins = [
        row["same_target_margin"]
        for row in summary_rows
        if row["position_family"] == "answer" and row.get("mean_same_mode_cosine", 0) >= row.get("mean_different_mode_cosine", 0)
    ]
    recommendation = {
        "best_target": query_best.get("target", best.get("target")),
        "best_position_family": query_best.get("position_family", best.get("position_family")),
        "best_layer": query_best.get("layer", best.get("layer")),
        f"best_top{config.top_k}_query_to_fact_recall": query_best.get(f"top{config.top_k}_query_to_fact_recall", 0.0),
        f"best_top{config.top_k}_same_target_recall": best.get(f"top{config.top_k}_same_target_recall", 0.0),
        "best_conflict_false_positive_rate": best.get("conflict_false_positive_rate", 1.0),
        "recommendation": _recommend(
            query_best.get(f"top{config.top_k}_query_to_fact_recall", best.get(f"top{config.top_k}_same_target_recall", 0.0)),
            best.get("conflict_false_positive_rate", 1.0),
            max(canonical_margins, default=0.0),
            config.top_k,
        ),
    }
    (output_dir / "recommendation.json").write_text(json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ResidualCache residual reuse.")
    parser.add_argument("--collect-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = analyze(AnalysisConfig(args.collect_dir, args.output_dir, args.top_k))
    print(f"Wrote analysis to {path}")


if __name__ == "__main__":
    main()
