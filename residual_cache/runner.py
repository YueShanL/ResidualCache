from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from residual_cache.analysis import AnalysisConfig, analyze
from residual_cache.data_process import DataConfig, build_dataset
from residual_cache.residual_collect import CollectConfig, POSITION_FAMILY_CHOICES, collect


@dataclass(frozen=True)
class PipelineConfig:
    model_name: str
    output_dir: Path
    max_facts: int = 12
    seed: int = 13
    max_new_tokens: int = 8
    dtype: str = "auto"
    device: str = "auto"
    limit: int | None = None
    top_k: int = 10
    position_family: str = "final_prompt"
    source: str = "synthetic"
    max_context_turns: int = 24
    convomem_root: Path | None = None
    convomem_chat_template: bool = False
    convomem_knowledge_prompt: bool = False


def run_pipeline(config: PipelineConfig) -> Path:
    dataset_path = config.output_dir / "data" / "pre_research_prompts.jsonl"
    collect_dir = config.output_dir / "collect"
    analysis_dir = config.output_dir / "analysis"
    build_dataset(
        DataConfig(
            output_path=dataset_path,
            max_facts=config.max_facts,
            seed=config.seed,
            source=config.source,
            max_context_turns=config.max_context_turns,
            convomem_root=config.convomem_root,
            convomem_chat_template=config.convomem_chat_template,
            convomem_knowledge_prompt=config.convomem_knowledge_prompt,
        )
    )
    collect(
        CollectConfig(
            model_name=config.model_name,
            dataset_path=dataset_path,
            output_dir=collect_dir,
            max_new_tokens=config.max_new_tokens,
            dtype=config.dtype,
            device=config.device,
            limit=config.limit,
            position_family=config.position_family,
        )
    )
    return analyze(AnalysisConfig(collect_dir, analysis_dir, config.top_k))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ResidualCache pre-research pipeline.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-facts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--position-family", choices=POSITION_FAMILY_CHOICES, default="final_prompt")
    parser.add_argument("--source", choices=("synthetic", "lufy", "convomem"), default="synthetic")
    parser.add_argument("--max-context-turns", type=int, default=24)
    parser.add_argument("--convomem-root", type=Path)
    parser.add_argument("--convomem-chat-template", action="store_true")
    parser.add_argument("--convomem-knowledge-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = run_pipeline(
        PipelineConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_facts=args.max_facts,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            device=args.device,
            limit=args.limit,
            top_k=args.top_k,
            position_family=args.position_family,
            source=args.source,
            max_context_turns=args.max_context_turns,
            convomem_root=args.convomem_root,
            convomem_chat_template=args.convomem_chat_template,
            convomem_knowledge_prompt=args.convomem_knowledge_prompt,
        )
    )
    print(f"Wrote pipeline analysis to {path}")


if __name__ == "__main__":
    main()
