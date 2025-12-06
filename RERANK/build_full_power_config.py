"""Utility to auto-generate a "full power" rerank config.

The script scans TRAIN/*/final_model (SFT baselines) and
RL/full_run*/ */final_model (RLHF variants) to build a single JSON config
that wires every available checkpoint into the reranking pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

BASE_GENERATION = {
    "device": "auto",
    "batch_size": 4,
    "max_input_length": 128,
    "max_new_tokens": 72,
    "num_beams": 4,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 1.08,
    "num_return_sequences": 2,
}

SFT_OVERRIDES = {
    "generation": {
        "do_sample": False,
        "num_beams": 6,
        "temperature": 1.0,
        "num_return_sequences": 1,
    }
}

RLFULL_OVERRIDES = {
    "generation": {
        "do_sample": True,
        "num_beams": 4,
        "top_p": 0.9,
        "temperature": 1.12,
        "num_return_sequences": 4,
    }
}

RLJONLY_OVERRIDES = {
    "generation": {
        "do_sample": True,
        "num_beams": 4,
        "top_p": 0.92,
        "temperature": 1.18,
        "num_return_sequences": 4,
    }
}

MODEL_BATCH_HINTS = {
    "bart-large": 2,
    "blenderbot": 2,
    "prophetnet-large-uncased": 2,
    "pure-t5-base": 2,
    "distilbart": 3,
}


def apply_batch_hint(entry: Dict, model_name: str) -> None:
    hint = MODEL_BATCH_HINTS.get(model_name)
    if hint is None:
        return
    current = entry.get("batch_size")
    entry["batch_size"] = hint if current is None else min(current, hint)


def collect_models(root: Path, tag: str, overrides: Dict | None = None) -> List[Dict]:
    overrides = overrides or {}
    entries: List[Dict] = []
    if not root.exists():
        print(f"[WARN] {root} not found; skipping {tag} family")
        return entries

    for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        final_dir = model_dir / "final_model"
        if not final_dir.is_dir():
            continue

        name = f"{model_dir.name}-{tag}"
        entry: Dict = {
            "name": name,
            "path": str(final_dir),
        }

        if overrides.get("batch_size"):
            entry["batch_size"] = overrides["batch_size"]

        apply_batch_hint(entry, model_dir.name)

        if overrides.get("generation"):
            entry["generation"] = dict(overrides["generation"])

        entries.append(entry)

    return entries


def build_config(args: argparse.Namespace) -> Dict:
    config = {
        "dataset_path": args.dataset_path,
        "source_column": args.source_column,
        "reference_column": args.reference_column,
        "max_samples": args.max_samples,
        "output_dir": args.output_dir,
        "generation": BASE_GENERATION,
        "models": [],
        "scoring": {
            "device": "auto",
            "batch_size": 32,
            "toxicity_threshold": 0.5,
            "objective_weights": {
                "J": 1.0,
                "BLEU_NORM": 0.25,
                "SIM": 0.1,
                "FL_PROB": 0.2,
            },
        },
    }

    families = [
        (Path("TRAIN"), "sft", SFT_OVERRIDES),
        (Path("RL/full_run"), "rlhf-jbleu", RLFULL_OVERRIDES),
        (Path("RL/full_run_jonly"), "rlhf-jonly", RLJONLY_OVERRIDES),
    ]

    for root, tag, overrides in families:
        models = collect_models(root, tag, overrides)
        config["models"].extend(models)
        print(f"[INFO] Added {len(models)} models from {root}")

    if not config["models"]:
        raise RuntimeError("No models discovered. Check your TRAIN/ and RL/ folders.")

    return config


def main():
    parser = argparse.ArgumentParser(description="Generate full-power rerank config")
    parser.add_argument("--output", default="RERANK/configs/full_power_config.json", help="Path to write config JSON")
    parser.add_argument("--dataset_path", default="TRAIN/paradetox.tsv")
    parser.add_argument("--source_column", default="toxic")
    parser.add_argument("--reference_column", default="neutral1")
    parser.add_argument("--output_dir", default="RERANK/outputs/full_power")
    parser.add_argument("--max_samples", default=None, type=int)

    args = parser.parse_args()
    config = build_config(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[DONE] Wrote config with {len(config['models'])} models -> {output_path}")


if __name__ == "__main__":
    main()
