"""Interactive CLI for full-power reranking.

Usage:
    python RERANK/rerank_cli.py --config RERANK/configs/rl_only_config.json \
        --input "<toxic sentence>" [--reference "<ref>"]

If --input is omitted, the script will prompt for user input. Multiple
sentences can be supplied via a TSV file (see --tsv option), but by default we
focus on single interactive queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from rerank_pipeline import CandidateGenerator, CandidateScorer, rerank_candidates


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_single_sample(toxic: str, reference: str | None = None) -> List[Dict[str, Any]]:
    sample = {
        "sample_id": 0,
        "source": toxic,
        "reference": reference or toxic,
    }
    return [sample]


def run_single_rerank(config: Dict[str, Any], toxic: str, reference: str | None = None):
    samples = build_single_sample(toxic, reference)

    generator = CandidateGenerator(config.get("models", []), config.get("generation", {}))
    candidates = generator.generate(samples)

    references = [samples[0]["reference"]]
    scorer = CandidateScorer(config.get("scoring", {}), [samples[0]["source"]])
    scored = scorer.score(candidates, references)
    best = rerank_candidates(scored)[0]

    return best


def main():
    parser = argparse.ArgumentParser(description="Interactive reranking demo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", dest="user_input", default=None, help="Toxic sentence to detoxify")
    parser.add_argument("--reference", default=None, help="Optional reference sentence for BLEU")
    parser.add_argument("--tsv", default=None, help="Optional TSV with toxic/ref columns to batch rerank")
    parser.add_argument("--source_column", default="toxic")
    parser.add_argument("--reference_column", default="neutral1")
    parser.add_argument("--generation_device", default=None, help="Force generation device (cpu/cuda/mps)")
    parser.add_argument("--scoring_device", default=None, help="Force scorer device (cpu/cuda/mps)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.generation_device:
        config.setdefault("generation", {})["device"] = args.generation_device
    if args.scoring_device:
        config.setdefault("scoring", {})["device"] = args.scoring_device

    if args.tsv:
        df = pd.read_csv(args.tsv, sep="\t")
        for idx, row in df.iterrows():
            toxic = row[args.source_column]
            reference = row.get(args.reference_column)
            print(f"\n===== Sample {idx} =====")
            result = run_single_rerank(config, toxic, reference)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    user_input = args.user_input or input("请输入需要降毒的句子：\n> ").strip()
    if not user_input:
        print("未输入任何内容，退出。")
        return

    best = run_single_rerank(config, user_input, args.reference)

    print("\n=== Rerank 结果 ===")
    print(f"候选文本: {best['text']}")
    print(f"来源模型: {best['model']} (rank #{best['candidate_rank']})")
    print(f"Objective分数: {best['objective']:.4f}")
    metrics = best["scores"]
    for key in ["J", "STA_PROB", "SIM", "FL_PROB", "BLEU_NORM"]:
        value = metrics.get(key)
        if value is not None:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()