from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    STATE_SCHEMA_VERSION,
    ClusterCandidate,
    DistributionState,
    EvaluationDataset,
    EvaluationExample,
    EvaluationModel,
    ModelRun,
)


STRATEGIES = (
    "full_context",
    "evidence_only",
    "local_only",
    "fixed_policy",
    "learned_router",
    "oracle_cluster",
)


@dataclass(frozen=True)
class ValidationRunConfig:
    budgets: tuple[int, ...] = (1, 2, 4, 8)
    fixed_policy: str = "recent"
    oracle_signal: str = "auto"
    maximum_samples: int | None = None
    continue_on_error: bool = False
    resume: bool = False

    def __post_init__(self) -> None:
        budgets = tuple(int(value) for value in self.budgets)
        if not budgets or any(value <= 0 for value in budgets):
            raise ValueError("budgets must contain positive integers")
        if budgets != tuple(sorted(set(budgets))):
            raise ValueError("budgets must be sorted and unique")
        object.__setattr__(self, "budgets", budgets)
        if self.fixed_policy != "recent":
            raise ValueError("the first validation runner supports fixed_policy='recent'")
        if self.oracle_signal not in {"auto", "teacher_attention", "evidence"}:
            raise ValueError("oracle_signal must be auto, teacher_attention, or evidence")
        if self.maximum_samples is not None and self.maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _condition_name(strategy: str, budget: int | None = None) -> str:
    return strategy if budget is None else f"{strategy}@{budget}"


def _group_by_layer(
    candidates: Sequence[ClusterCandidate],
) -> dict[int, list[ClusterCandidate]]:
    grouped: dict[int, list[ClusterCandidate]] = {}
    identities: set[tuple[int, str]] = set()
    records: set[tuple[int, str]] = set()
    for candidate in candidates:
        identity = (candidate.layer, candidate.cluster_id)
        if identity in identities:
            raise ValueError(f"duplicate cluster candidate: {identity}")
        identities.add(identity)
        for record_id in candidate.record_ids:
            record = (candidate.layer, record_id)
            if record in records:
                raise ValueError(f"record appears in multiple current leaves: {record}")
            records.add(record)
        grouped.setdefault(candidate.layer, []).append(candidate)
    return grouped


def _oracle_score(
    candidate: ClusterCandidate,
    *,
    signal: str,
    layer_has_teacher: bool,
) -> float:
    use_teacher = signal == "teacher_attention" or (
        signal == "auto" and layer_has_teacher
    )
    if use_teacher:
        if candidate.teacher_attention_mass is None:
            return 0.0
        return candidate.teacher_attention_mass
    return float(candidate.evidence_token_count)


def select_clusters(
    candidates: Sequence[ClusterCandidate],
    *,
    strategy: str,
    budget: int,
    oracle_signal: str = "auto",
) -> dict[int, tuple[str, ...]]:
    """Apply the three built-in cluster policies independently per layer."""

    if strategy not in {"fixed_policy", "learned_router", "oracle_cluster"}:
        raise ValueError(f"unsupported cluster strategy: {strategy}")
    if budget <= 0:
        raise ValueError("budget must be positive")
    selected: dict[int, tuple[str, ...]] = {}
    for layer, rows in sorted(_group_by_layer(candidates).items()):
        layer_has_teacher = any(row.teacher_attention_mass is not None for row in rows)
        scored: list[tuple[float, ClusterCandidate]] = []
        for row in rows:
            if strategy == "fixed_policy":
                score = float(row.latest_position)
            elif strategy == "learned_router":
                if row.learned_log_score is None and row.learned_probability is None:
                    continue
                score = (
                    row.learned_log_score
                    if row.learned_log_score is not None
                    else float(row.learned_probability)
                )
            else:
                score = _oracle_score(
                    row,
                    signal=oracle_signal,
                    layer_has_teacher=layer_has_teacher,
                )
            scored.append((float(score), row))
        scored.sort(key=lambda item: (-item[0], item[1].cluster_id))
        selected[layer] = tuple(row.cluster_id for _score, row in scored[:budget])
    return selected


def _selected_candidates(
    candidates: Sequence[ClusterCandidate],
    selection: Mapping[int, Sequence[str]],
) -> list[ClusterCandidate]:
    by_id = {(row.layer, row.cluster_id): row for row in candidates}
    selected: list[ClusterCandidate] = []
    for layer, cluster_ids in selection.items():
        for cluster_id in cluster_ids:
            try:
                selected.append(by_id[(int(layer), str(cluster_id))])
            except KeyError as error:
                raise KeyError(
                    f"selection references unknown cluster {(layer, cluster_id)!r}"
                ) from error
    return selected


def _selection_state(
    example: EvaluationExample,
    candidates: Sequence[ClusterCandidate],
    selection: Mapping[int, Sequence[str]],
) -> dict[str, Any]:
    chosen = _selected_candidates(candidates, selection)
    selected_records = {
        (row.layer, record_id) for row in chosen for record_id in row.record_ids
    }
    selected_evidence_blocks = {
        block_id for row in chosen for block_id in row.evidence_block_ids
    }
    candidate_evidence_blocks = {
        block_id for row in candidates for block_id in row.evidence_block_ids
    }
    total_evidence_blocks = set(example.evidence_block_ids) or candidate_evidence_blocks
    teacher_available = any(
        row.teacher_attention_mass is not None for row in candidates
    )
    per_layer_tokens: dict[int, int] = {}
    for row in chosen:
        per_layer_tokens[row.layer] = (
            per_layer_tokens.get(row.layer, 0) + row.record_token_count
        )
    return {
        "selected_cluster_ids_by_layer": {
            str(layer): list(cluster_ids)
            for layer, cluster_ids in sorted(selection.items())
        },
        "selected_cluster_count": len(chosen),
        "selected_record_count": len(selected_records),
        "selected_record_token_count": sum(row.record_token_count for row in chosen),
        "selected_evidence_record_count": sum(
            row.evidence_record_count for row in chosen
        ),
        "selected_evidence_token_count": sum(
            row.evidence_token_count for row in chosen
        ),
        "total_cluster_evidence_record_count": sum(
            row.evidence_record_count for row in candidates
        ),
        "total_cluster_evidence_token_count": sum(
            row.evidence_token_count for row in candidates
        ),
        "dataset_evidence_token_count": example.evidence_token_count,
        "selected_evidence_block_ids": sorted(selected_evidence_blocks),
        "total_evidence_block_ids": sorted(total_evidence_blocks),
        "selected_teacher_attention_mass": (
            None
            if not teacher_available
            else sum(float(row.teacher_attention_mass or 0.0) for row in chosen)
        ),
        "total_teacher_attention_mass": (
            None
            if not teacher_available
            else sum(float(row.teacher_attention_mass or 0.0) for row in candidates)
        ),
        "mean_selected_cluster_record_count": (
            0.0
            if not chosen
            else sum(len(row.record_ids) for row in chosen) / len(chosen)
        ),
        "per_layer_retrieved_tokens": {
            str(layer): count for layer, count in sorted(per_layer_tokens.items())
        },
    }


def _condition_state(
    *,
    name: str,
    strategy: str,
    budget: int | None,
    run: ModelRun,
    distribution: DistributionState | None,
    selection_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "strategy": strategy,
        "budget": budget,
        "prediction": {
            "text": run.predicted_text,
            "token_ids": list(run.predicted_token_ids),
        },
        "distribution": None if distribution is None else distribution.state_dict(),
        "selection": None if selection_state is None else dict(selection_state),
        "resources": run.resources.state_dict(),
        "state": dict(run.state),
    }


def _manifest_identity(
    dataset: EvaluationDataset,
    model: EvaluationModel,
    config: ValidationRunConfig,
) -> dict[str, Any]:
    payload = {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "dataset": dict(dataset.descriptor),
        "model": dict(model.descriptor),
        "run_config": asdict(config) | {"resume": False},
        "strategies": list(STRATEGIES),
    }
    payload["identity_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _completed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                sample_id = str(row["sample"]["sample_id"])
                if sample_id in result:
                    raise ValueError(f"duplicate sample in state file: {sample_id}")
                result.add(sample_id)
    return result


def collect_validation_states(
    dataset: EvaluationDataset,
    model: EvaluationModel,
    output_dir: str | Path,
    config: ValidationRunConfig | None = None,
) -> dict[str, Any]:
    """Run all comparison policies and stream versioned per-sample state."""

    config = config or ValidationRunConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    samples_path = output_dir / "samples.jsonl"
    errors_path = output_dir / "errors.jsonl"
    identity = _manifest_identity(dataset, model, config)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not config.resume:
            raise FileExistsError(
                f"validation state already exists; enable resume: {manifest_path}"
            )
        if existing.get("identity_sha256") != identity["identity_sha256"]:
            raise ValueError("existing validation state identity does not match this run")
    elif samples_path.exists() or errors_path.exists():
        raise FileExistsError("state files exist without a matching run manifest")
    else:
        _atomic_json(
            manifest_path,
            identity
            | {
                "status": "running",
                "started_at_unix": time.time(),
                "completed_sample_count": 0,
                "error_count": 0,
            },
        )

    completed = _completed_sample_ids(samples_path)
    completed_count = len(completed)
    error_count = (
        0
        if not errors_path.exists()
        else sum(
            1
            for line in errors_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    yielded_count = 0
    mode = "a" if samples_path.exists() else "x"
    error_mode = "a" if errors_path.exists() else "x"
    with samples_path.open(mode, encoding="utf-8", newline="\n") as sample_handle, errors_path.open(
        error_mode, encoding="utf-8", newline="\n"
    ) as error_handle:
        seen_dataset_ids: set[str] = set()
        for example in dataset:
            if example.sample_id in seen_dataset_ids:
                raise ValueError(f"dataset emitted duplicate sample_id: {example.sample_id}")
            seen_dataset_ids.add(example.sample_id)
            if example.sample_id in completed:
                continue
            if config.maximum_samples is not None and yielded_count >= config.maximum_samples:
                break
            yielded_count += 1
            session = None
            try:
                session = model.open(example)
                candidates = tuple(session.cluster_candidates())
                _group_by_layer(candidates)
                reference = session.run_full_context()
                evidence_only = session.run_evidence_only()
                local = session.run_local_only()
                conditions: dict[str, Any] = {}
                for strategy, run in (
                    ("full_context", reference),
                    ("evidence_only", evidence_only),
                    ("local_only", local),
                ):
                    distribution = session.compact_distribution(reference, run)
                    conditions[strategy] = _condition_state(
                        name=strategy,
                        strategy=strategy,
                        budget=None,
                        run=run,
                        distribution=distribution,
                        selection_state=None,
                    )
                for budget in config.budgets:
                    for strategy in (
                        "fixed_policy",
                        "learned_router",
                        "oracle_cluster",
                    ):
                        selection = select_clusters(
                            candidates,
                            strategy=strategy,
                            budget=budget,
                            oracle_signal=config.oracle_signal,
                        )
                        selection_state = _selection_state(
                            example, candidates, selection
                        )
                        run = session.run_with_clusters(
                            selection,
                            strategy=strategy,
                            budget=budget,
                        )
                        selected_tokens = {
                            int(layer): int(count)
                            for layer, count in selection_state[
                                "per_layer_retrieved_tokens"
                            ].items()
                        }
                        run.resources = run.resources.with_default_historical_tokens(
                            selected_tokens
                        )
                        name = _condition_name(strategy, budget)
                        conditions[name] = _condition_state(
                            name=name,
                            strategy=strategy,
                            budget=budget,
                            run=run,
                            distribution=session.compact_distribution(reference, run),
                            selection_state=selection_state,
                        )
                row = {
                    "state_schema_version": STATE_SCHEMA_VERSION,
                    "sample": example.state_dict(),
                    "candidates": [candidate.state_dict() for candidate in candidates],
                    "conditions": conditions,
                }
                sample_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                sample_handle.flush()
                completed.add(example.sample_id)
                completed_count += 1
            except Exception as error:
                error_count += 1
                error_handle.write(
                    json.dumps(
                        {
                            "sample_id": example.sample_id,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_handle.flush()
                if not config.continue_on_error:
                    _atomic_json(
                        manifest_path,
                        identity
                        | {
                            "status": "failed",
                            "completed_sample_count": completed_count,
                            "error_count": error_count,
                            "updated_at_unix": time.time(),
                        },
                    )
                    raise
            finally:
                close = getattr(session, "close", None)
                if close is not None:
                    close()

    manifest = identity | {
        "status": "complete" if error_count == 0 else "complete_with_errors",
        "completed_sample_count": completed_count,
        "error_count": error_count,
        "completed_at_unix": time.time(),
        "samples_file": samples_path.name,
        "errors_file": errors_path.name,
    }
    _atomic_json(manifest_path, manifest)
    return manifest
