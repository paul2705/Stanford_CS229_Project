import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sacrebleu
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------

def resolve_device(requested: str | None = None) -> torch.device:
    """Pick an available torch device, honoring user preference when possible."""

    if requested:
        req = requested.lower()
        if req.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        if req == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if req == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_json_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Candidate generation
# ------------------------------------------------------------


class CandidateGenerator:
    def __init__(self, model_configs: List[Dict[str, Any]], generation_cfg: Dict[str, Any]):
        if not model_configs:
            raise ValueError("At least one model must be specified for candidate generation.")

        self.model_configs = model_configs
        self.global_cfg = generation_cfg or {}
        self.device = resolve_device(self.global_cfg.get("device"))
        self.default_batch_size = self.global_cfg.get("batch_size", 8)
        self.default_max_input = self.global_cfg.get("max_input_length", 128)

    def _load_model(self, cfg: Dict[str, Any]):
        model_path = cfg["path"]
        tokenizer_name = cfg.get("tokenizer") or model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def _build_generate_kwargs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.global_cfg)
        merged.update(cfg.get("generation", {}))

        kwargs: Dict[str, Any] = {
            "max_new_tokens": merged.get("max_new_tokens", 60),
            "num_beams": merged.get("num_beams", 4),
            "do_sample": merged.get("do_sample", False),
            "temperature": merged.get("temperature", 1.0),
            "top_p": merged.get("top_p", 0.92),
            "top_k": merged.get("top_k", 50),
            "num_return_sequences": merged.get("num_return_sequences", 1),
            "early_stopping": merged.get("early_stopping", True),
            "length_penalty": merged.get("length_penalty", 1.0),
        }

        if not kwargs["do_sample"]:
            # Top-k / top-p only matter for sampling. Remove to avoid HF warnings.
            kwargs.pop("top_p", None)
            kwargs.pop("top_k", None)

        return kwargs

    def generate(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run each configured model to produce candidate rewrites for every sample."""

        all_candidates: List[Dict[str, Any]] = []
        if not samples:
            return all_candidates

        for cfg in self.model_configs:
            name = cfg.get("name") or Path(cfg["path"]).name
            tokenizer, model = self._load_model(cfg)
            gen_kwargs = self._build_generate_kwargs(cfg)
            batch_size = cfg.get("batch_size", self.default_batch_size)
            max_input = cfg.get("max_input_length", self.default_max_input)
            num_return = gen_kwargs.get("num_return_sequences", 1)

            print(f"\n[Generator] {name}: device={self.device.type}, batch={batch_size}, returns={num_return}")
            for start in tqdm(range(0, len(samples), batch_size), desc=f"Generating ({name})"):
                batch = samples[start : start + batch_size]
                prompts = [item["source"] for item in batch]
                enc = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=max_input,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **enc,
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        num_beams=gen_kwargs["num_beams"],
                        do_sample=gen_kwargs["do_sample"],
                        temperature=gen_kwargs.get("temperature", 1.0),
                        top_p=gen_kwargs.get("top_p"),
                        top_k=gen_kwargs.get("top_k"),
                        num_return_sequences=num_return,
                        early_stopping=gen_kwargs.get("early_stopping", True),
                        length_penalty=gen_kwargs.get("length_penalty", 1.0),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for idx, text in enumerate(decoded):
                    prompt_offset = idx // num_return
                    candidate_rank = idx % num_return
                    sample = batch[prompt_offset]
                    all_candidates.append(
                        {
                            "sample_id": sample["sample_id"],
                            "model": name,
                            "candidate_rank": candidate_rank,
                            "text": text.strip(),
                        }
                    )

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return all_candidates


# ------------------------------------------------------------
# Candidate scoring (ParaDetox mini-evaluator)
# ------------------------------------------------------------


class CandidateScorer:
    def __init__(self, scoring_cfg: Dict[str, Any], sources: List[str]):
        self.cfg = scoring_cfg or {}
        self.device = resolve_device(self.cfg.get("device"))
        self.batch_size = self.cfg.get("batch_size", 32)
        self.tox_threshold = self.cfg.get("toxicity_threshold", 0.5)

        tox_model_name = self.cfg.get("toxicity_model", "SkolkovoInstitute/roberta_toxicity_classifier")
        cola_model_name = self.cfg.get("cola_model", "textattack/roberta-base-CoLA")
        sim_model_name = self.cfg.get("similarity_model", "sentence-transformers/all-mpnet-base-v2")

        print(f"[Scorer] Loading toxicity classifier: {tox_model_name}")
        self.tox_tokenizer = AutoTokenizer.from_pretrained(tox_model_name)
        self.tox_model = AutoModelForSequenceClassification.from_pretrained(tox_model_name).to(self.device).eval()

        print(f"[Scorer] Loading CoLA classifier: {cola_model_name}")
        self.cola_tokenizer = AutoTokenizer.from_pretrained(cola_model_name)
        self.cola_model = AutoModelForSequenceClassification.from_pretrained(cola_model_name).to(self.device).eval()

        sim_device = self.cfg.get("similarity_device") or self.device.type
        print(f"[Scorer] Loading SentenceTransformer: {sim_model_name} ({sim_device})")
        self.sim_model = SentenceTransformer(sim_model_name, device=sim_device)

        self.source_embeddings = self.sim_model.encode(
            sources,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        weight_cfg = self.cfg.get("objective_weights") or {"J": 1.0}
        self.objective_weights = {k.upper(): float(v) for k, v in weight_cfg.items()}

    def _batch_iter(self, items: List[str]):
        for i in range(0, len(items), self.batch_size):
            yield items[i : i + self.batch_size]

    def _score_toxicity(self, sentences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        probs_non_toxic: List[float] = []
        binary: List[int] = []
        with torch.no_grad():
            for batch in self._batch_iter(sentences):
                enc = self.tox_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.tox_model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                p_toxic = probs[:, 1]
                p_non_toxic = 1 - p_toxic
                probs_non_toxic.extend(p_non_toxic.cpu().numpy())
                binary.extend((p_non_toxic >= (1 - self.tox_threshold)).int().cpu().tolist())
        return np.array(probs_non_toxic, dtype=np.float32), np.array(binary, dtype=np.int32)

    def _score_fluency(self, sentences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        probs_accept: List[float] = []
        binary: List[int] = []
        with torch.no_grad():
            for batch in self._batch_iter(sentences):
                enc = self.cola_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.cola_model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                p_accept = probs[:, 1]
                probs_accept.extend(p_accept.cpu().numpy())
                binary.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        return np.array(probs_accept, dtype=np.float32), np.array(binary, dtype=np.int32)

    def score(self, candidates: List[Dict[str, Any]], references: List[str]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        sta_prob, sta_binary = self._score_toxicity(texts)
        fl_prob, fl_binary = self._score_fluency(texts)
        cand_embeddings = self.sim_model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        scored: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            sample_id = cand["sample_id"]
            sim = float(np.sum(cand_embeddings[idx] * self.source_embeddings[sample_id]))
            bleu = sacrebleu.sentence_bleu(cand["text"], [references[sample_id]]).score
            bleu_norm = bleu / 100.0
            j_prob = float(sta_prob[idx] * sim * fl_prob[idx])
            j_binary = float(sta_binary[idx] * sim * fl_binary[idx])

            metrics_map = {
                "STA": float(sta_binary[idx]),
                "STA_PROB": float(sta_prob[idx]),
                "FL": float(fl_binary[idx]),
                "FL_PROB": float(fl_prob[idx]),
                "SIM": sim,
                "BLEU": bleu,
                "BLEU_NORM": bleu_norm,
                "J": j_prob,
                "J_BINARY": j_binary,
            }

            objective = 0.0
            for metric_name, weight in self.objective_weights.items():
                value = metrics_map.get(metric_name)
                if value is None:
                    raise ValueError(f"Metric '{metric_name}' not available for objective scoring.")
                objective += weight * value

            scored.append(
                {
                    **cand,
                    "scores": metrics_map,
                    "objective": float(objective),
                }
            )

        return scored


# ------------------------------------------------------------
# Reranking + aggregation
# ------------------------------------------------------------


def rerank_candidates(scored_candidates: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    best: Dict[int, Dict[str, Any]] = {}
    for cand in scored_candidates:
        sample_id = cand["sample_id"]
        if sample_id not in best or cand["objective"] > best[sample_id]["objective"]:
            best[sample_id] = cand
    return best


def summarize_metrics(
    selected: Dict[int, Dict[str, Any]],
    samples: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    outputs: List[str] = []
    references: List[str] = []
    sta_vals: List[int] = []
    fl_vals: List[int] = []
    sim_vals: List[float] = []

    for sample in samples:
        candidate = selected.get(sample["sample_id"])
        if not candidate:
            continue
        outputs.append(candidate["text"])
        references.append(sample["reference"])
        sta_vals.append(int(round(candidate["scores"]["STA"])))
        fl_vals.append(int(round(candidate["scores"]["FL"])))
        sim_vals.append(candidate["scores"]["SIM"])

    if not outputs:
        return {}, outputs, references

    bleu = sacrebleu.corpus_bleu(outputs, [references]).score
    sta = float(np.mean(sta_vals))
    fl = float(np.mean(fl_vals))
    sim = float(np.mean(sim_vals))
    j = sta * sim * fl

    metrics = {
        "BLEU": bleu,
        "STA": sta,
        "SIM": sim,
        "FL": fl,
        "J": j,
    }
    return metrics, outputs, references


# ------------------------------------------------------------
# Main orchestration
# ------------------------------------------------------------


def run_pipeline(config: Dict[str, Any]):
    dataset_path = config.get("dataset_path") or "TRAIN/paradetox.tsv"
    source_col = config.get("source_column", "toxic")
    reference_col = config.get("reference_column", "neutral1")
    max_samples = config.get("max_samples")
    sample_strategy = (config.get("sample_strategy") or "head").lower()
    sample_seed = config.get("sample_seed")
    output_dir = Path(config.get("output_dir", "RERANK/outputs/latest"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline] Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, sep="\t")
    df = df.dropna(subset=[source_col, reference_col]).reset_index(drop=True)
    if max_samples:
        count = int(max_samples)
        if sample_strategy == "random":
            df = df.sample(n=count, random_state=sample_seed).reset_index(drop=True)
            print(f"[Pipeline] Randomly sampled {len(df)} rows (seed={sample_seed})")
        else:
            df = df.head(count)
            print(f"[Pipeline] Truncated dataset to first {len(df)} samples")

    samples = [
        {
            "sample_id": idx,
            "source": row[source_col],
            "reference": row[reference_col],
        }
        for idx, row in df.iterrows()
    ]

    generator = CandidateGenerator(config.get("models", []), config.get("generation", {}))
    candidates = generator.generate(samples)
    print(f"[Pipeline] Generated {len(candidates)} total candidates")

    references = [sample["reference"] for sample in samples]
    scorer = CandidateScorer(config.get("scoring", {}), [sample["source"] for sample in samples])
    scored_candidates = scorer.score(candidates, references)
    print(f"[Pipeline] Scored {len(scored_candidates)} candidates")

    selected = rerank_candidates(scored_candidates)
    print(f"[Pipeline] Selected best candidates for {len(selected)} samples")

    metrics, outputs, references_ordered = summarize_metrics(selected, samples)
    if not metrics:
        print("[Pipeline] No outputs selected; aborting save step.")
        return

    print("\n[Pipeline] Final aggregated metrics:")
    for key, value in metrics.items():
        if key == "BLEU":
            print(f"  {key}: {value:6.2f}")
        else:
            print(f"  {key}: {value:6.3f}")

    candidates_df = pd.DataFrame(
        [
            {
                "sample_id": cand["sample_id"],
                "model": cand["model"],
                "candidate_rank": cand["candidate_rank"],
                "text": cand["text"],
                "objective": cand["objective"],
                **cand["scores"],
            }
            for cand in scored_candidates
        ]
    )
    candidates_path = output_dir / "all_candidates.tsv"
    candidates_df.to_csv(candidates_path, sep="\t", index=False)

    selected_rows = []
    for sample in samples:
        cand = selected.get(sample["sample_id"])
        if not cand:
            continue
        selected_rows.append(
            {
                "sample_id": sample["sample_id"],
                "toxic": sample["source"],
                "reference": sample["reference"],
                "model_output": cand["text"],
                "chosen_model": cand["model"],
                "candidate_rank": cand["candidate_rank"],
                "objective": cand["objective"],
                "sta_prob": cand["scores"]["STA_PROB"],
                "sta": cand["scores"]["STA"],
                "sim": cand["scores"]["SIM"],
                "fl_prob": cand["scores"]["FL_PROB"],
                "fl": cand["scores"]["FL"],
                "bleu": cand["scores"]["BLEU"],
                "j_prob": cand["scores"]["J"],
            }
        )
    selected_df = pd.DataFrame(selected_rows)
    selected_path = output_dir / "reranked_outputs.tsv"
    selected_df.to_csv(selected_path, sep="\t", index=False)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    config_out = output_dir / "config.used.json"
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n[Pipeline] Saved candidates -> {candidates_path}")
    print(f"[Pipeline] Saved reranked outputs -> {selected_path}")
    print(f"[Pipeline] Saved metrics -> {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model reranking pipeline for detoxification")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config describing dataset, models, generation, and scoring settings.",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Optional JSON string to override config keys (e.g., '{\"max_samples\": 200}').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    if args.override:
        overrides = json.loads(args.override)
        config.update(overrides)
    run_pipeline(config)


if __name__ == "__main__":
    main()
