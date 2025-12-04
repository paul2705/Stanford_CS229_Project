import argparse
import contextlib
import json
import os
import random
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
import sacrebleu
from tqdm.auto import tqdm

try:
	from trl import AutoModelForSeq2SeqLMWithValueHead
except ImportError:
	raise ImportError("Please install trl before running RLHF training (pip install trl)")

try:
	from detoxify import Detoxify
	from sentence_transformers import SentenceTransformer, util as st_util
except ImportError as exc:
	raise ImportError("Detoxify and sentence-transformers are required for reward modeling") from exc


# Allow re-use of existing helpers and metric scripts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.append(ROOT_DIR)

from TRAIN.train_detox import prepare_data  # noqa: E402
from METRIC.evaluate_paradetox_metrics import evaluate_paradetox  # noqa: E402


warnings.filterwarnings("ignore")


def parse_args():
	"""Parse CLI arguments for RLHF training and evaluation."""
	parser = argparse.ArgumentParser(description="RLHF training entry point")
	parser.add_argument("--model_name", type=str, default="facebook/bart-base",
						help="HF checkpoint used when no supervised model is provided")
	parser.add_argument("--data_path", type=str, default="./TRAIN/paradetox.tsv",
						help="Path to ParaDetox TSV input")
	parser.add_argument("--input_dir", type=str, default="./TRAIN/bart-base",
						help="Directory containing a supervised fine-tuned model")
	parser.add_argument("--output_dir", type=str, default="./RL/bart-base",
						help="Directory to save RLHF checkpoints and reports")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")

	# RL subset sizes
	parser.add_argument("--rl_train_size", type=int, default=8000,
						help="Number of prompts used for PPO training")
	parser.add_argument("--rl_eval_size", type=int, default=1000,
						help="Number of prompts used for RL evaluation")

	# PPO hyperparameters
	parser.add_argument("--ppo_epochs", type=int, default=3, help="Number of PPO epochs")
	parser.add_argument("--ppo_steps_per_epoch", type=int, default=10,
						help="PPO updates per epoch")
	parser.add_argument("--ppo_batch_size", type=int, default=8, help="Prompt batch size for PPO")
	parser.add_argument("--ppo_lr", type=float, default=8e-6, help="Learning rate for PPO")
	parser.add_argument("--ppo_clip_range", type=float, default=0.15, help="Policy clip range")
	parser.add_argument("--ppo_kl_target", type=float, default=0.3, help="Target KL divergence")
	parser.add_argument("--ppo_kl_rate", type=float, default=0.1, help="KL coefficient adjustment rate")
	parser.add_argument("--ppo_kl_init", type=float, default=0.2, help="Initial KL coefficient")

	# Reward settings
	parser.add_argument("--reward_alpha", type=float, default=0.7,
						help="Weight for toxicity reward")
	parser.add_argument("--reward_beta", type=float, default=0.3,
						help="Weight for semantic reward")
	parser.add_argument("--reward_min_len", type=int, default=3,
						help="Minimum token count to avoid short-output penalty")
	parser.add_argument("--metric_reward_weight", type=float, default=0.6,
					help="Blend weight for ParaDetox J-based reward (0 disables)")
	parser.add_argument("--metric_reward_stride", type=int, default=1,
					help="Inject metric reward every N PPO steps")
	parser.add_argument("--bleu_reward_weight", type=float, default=0.0,
					help="Additional weight for sentence-level BLEU reward (0 disables)")
	parser.add_argument("--bleu_reward_stride", type=int, default=1,
					help="Inject BLEU reward every N PPO steps")

	# Generation + evaluation
	parser.add_argument("--max_input_length", type=int, default=128,
						help="Maximum prompt length")
	parser.add_argument("--max_new_tokens", type=int, default=60,
						help="Maximum tokens generated per response")
	parser.add_argument("--eval_sample_size", type=int, default=200,
						help="Number of prompts for periodic evaluation")
	parser.add_argument("--inference_batch_size", type=int, default=32,
						help="Batch size for held-out generation/metrics")
	parser.add_argument("--fp16_generation", dest="fp16_generation", action="store_true",
						help="Force-enable autocast for generation on accelerator devices")
	parser.add_argument("--no_fp16_generation", dest="fp16_generation", action="store_false",
						help="Force-disable autocast for generation")
	parser.set_defaults(fp16_generation=None)

	return parser.parse_args()


def autocast_context(device: torch.device, enabled: bool):
	"""Return a context manager that enables autocast when supported."""
	if enabled and device.type in {"cuda", "mps"}:
		return torch.autocast(device_type=device.type, dtype=torch.float16)
	return contextlib.nullcontext()


def set_seed(seed: int):
	"""Seed python, numpy, and torch RNGs for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)  # keep multi-GPU runs deterministic


def resolve_device() -> torch.device:
	"""Choose the best available accelerator, preferring CUDA then MPS."""
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def read_data(path: str) -> pd.DataFrame:
	"""Load the ParaDetox TSV input as a pandas DataFrame."""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Data file not found: {path}")
	return pd.read_csv(path, sep="\t")


def split_dataset(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Split processed pairs into train/validation/test partitions."""
	train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
	val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
	return (train_df.reset_index(drop=True),
			val_df.reset_index(drop=True),
			test_df.reset_index(drop=True))


class PromptDataset(Dataset):
	"""Simple Dataset wrapper that exposes detox prompts + references for RL."""

	def __init__(self, prompts: List[str], references: List[str]):
		if len(prompts) != len(references):
			raise ValueError("Prompts and references must have identical lengths")
		self.prompts = list(prompts)
		self.references = list(references)

	def __len__(self):
		"""Return dataset length expected by DataLoader."""
		return len(self.prompts)

	def __getitem__(self, idx):
		"""Return the prompt/reference pair at a specific index."""
		return {
			"prompt": self.prompts[idx],
			"reference": self.references[idx],
		}


def collate_prompts(batch: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int):
	"""Tokenize a batch of prompts and attach raw/reference text for reward computation."""
	prompts = [item["prompt"] for item in batch]
	refs = [item["reference"] for item in batch]
	tokenized = tokenizer(
		prompts,
		max_length=max_length,
		truncation=True,
		padding=True,
		return_tensors="pt",
	)
	tokenized["prompt_text"] = prompts
	tokenized["reference_text"] = refs
	return tokenized


def build_prompt_loader(prompts: List[str], references: List[str], tokenizer: AutoTokenizer,
					batch_size: int, max_length: int, shuffle: bool) -> DataLoader:
	"""Create a DataLoader that streams prompt/reference pairs for PPO steps."""
	dataset = PromptDataset(prompts, references)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		collate_fn=lambda batch: collate_prompts(batch, tokenizer, max_length),
	)


def ensure_dir(path: str):
	"""Create a directory if it does not exist."""
	os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str):
	"""Persist a Python dict as formatted JSON."""
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, indent=2, ensure_ascii=False)


def model_exists(model_dir: str) -> bool:
	"""Check whether a saved HF model (config.json) exists in a directory."""
	config_path = os.path.join(model_dir, "config.json")
	return os.path.isfile(config_path)


def resolve_base_model_path(input_dir: str, model_name: str) -> str:
	"""Return supervised checkpoint path or fallback to raw HF model."""
	candidate = os.path.join(input_dir, "final_model")
	if model_exists(candidate):
		return candidate
	warnings.warn(
		f"Could not find supervised checkpoint under {candidate}. Falling back to {model_name}."
	)
	return model_name


def export_config(args, output_dir: str, extra: Dict):
	"""Dump runtime configuration and metadata for experiment tracking."""
	cfg = vars(args).copy()
	cfg.update(extra)
	save_json(cfg, os.path.join(output_dir, "config.json"))  # mirror TRAIN logging style


class RewardScorer:
	"""Encapsulates Detoxify toxicity and Sentence-BERT semantic scoring."""

	def __init__(self, device: torch.device):
		device_str = str(device)
		if device.type == "mps":
			device_str = "cpu"  # Detoxify GPU kernels do not support MPS
		self.tox_model = Detoxify("original", device=device_str)
		self.sim_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device_str)

	def score_toxicity(self, texts: List[str]) -> np.ndarray:
		"""Return toxicity probabilities for generated texts."""
		results = self.tox_model.predict(texts)
		return np.array(results["toxicity"], dtype=np.float32)

	def score_similarity(self, prompts: List[str], outputs: List[str]) -> np.ndarray:
		"""Compute cosine similarity between prompt/output embeddings."""
		prompt_emb = self.sim_model.encode(prompts, convert_to_tensor=True, show_progress_bar=False)
		output_emb = self.sim_model.encode(outputs, convert_to_tensor=True, show_progress_bar=False)
		sims = st_util.cos_sim(prompt_emb, output_emb).diag().cpu().numpy()
		return np.clip(sims, -1.0, 1.0)

	def compute_rewards(self, prompts: List[str], outputs: List[str],
						alpha: float, beta: float, min_len: int) -> np.ndarray:
		"""Blend toxicity and similarity into a bounded scalar reward."""
		tox_scores = self.score_toxicity(outputs)
		sim_scores = self.score_similarity(prompts, outputs)
		rewards = alpha * (1.0 - tox_scores) + beta * sim_scores
		for idx, text in enumerate(outputs):
			if len(text.strip().split()) < min_len or text.strip() == prompts[idx].strip():
				rewards[idx] -= 0.2  # discourage trivial or too-short outputs
		return np.clip(rewards, -1.0, 1.0)


class ParaDetoxMiniEvaluator:
	"""Lightweight evaluator that reproduces ParaDetox STA/SIM/FL for mini batches."""

	TOX_MODEL = "SkolkovoInstitute/roberta_toxicity_classifier"
	COLA_MODEL = "textattack/roberta-base-CoLA"
	SIM_MODEL = "sentence-transformers/all-mpnet-base-v2"

	def __init__(self, device: torch.device, batch_size: int = 16):
		metric_device = device if device.type in {"cuda", "mps"} else torch.device("cpu")
		self.device = metric_device
		self.batch_size = batch_size
		self.tox_tokenizer = AutoTokenizer.from_pretrained(self.TOX_MODEL)
		self.tox_model = AutoModelForSequenceClassification.from_pretrained(self.TOX_MODEL).to(metric_device).eval()
		self.cola_tokenizer = AutoTokenizer.from_pretrained(self.COLA_MODEL)
		self.cola_model = AutoModelForSequenceClassification.from_pretrained(self.COLA_MODEL).to(metric_device).eval()
		self.sim_model = SentenceTransformer(self.SIM_MODEL, device=str(metric_device))

	def _batched(self, items: List[str]):
		for start in range(0, len(items), self.batch_size):
			yield items[start:start + self.batch_size]

	def _classifier_prob(self, texts: List[str], tokenizer, model, positive_index: int) -> np.ndarray:
		probs = []
		with torch.no_grad():
			for batch in self._batched(texts):
				enc = tokenizer(
					batch,
					padding=True,
					truncation=True,
					max_length=128,
					return_tensors="pt",
				).to(self.device)
				logits = model(**enc).logits
				p = torch.softmax(logits, dim=-1)[:, positive_index]
				probs.extend(p.detach().cpu().numpy())
		return np.array(probs, dtype=np.float32)

	def compute_sta(self, outputs: List[str]) -> np.ndarray:
		"""Return per-sample non-toxicity probability."""
		tox_probs = self._classifier_prob(outputs, self.tox_tokenizer, self.tox_model, positive_index=1)
		sta = 1.0 - tox_probs
		return np.clip(sta, 0.0, 1.0)

	def compute_fl(self, outputs: List[str]) -> np.ndarray:
		"""Return per-sample CoLA acceptability probability."""
		fl = self._classifier_prob(outputs, self.cola_tokenizer, self.cola_model, positive_index=1)
		return np.clip(fl, 0.0, 1.0)

	def compute_sim(self, sources: List[str], outputs: List[str]) -> np.ndarray:
		"""Return cosine similarity mapped to [0, 1]."""
		src_emb = self.sim_model.encode(
			sources,
			batch_size=self.batch_size,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		)
		out_emb = self.sim_model.encode(
			outputs,
			batch_size=self.batch_size,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		)
		sims = np.sum(src_emb * out_emb, axis=1)
		sims = np.clip(sims, -1.0, 1.0)
		return 0.5 * (sims + 1.0)

	def compute_sentence_bleu(self, references: List[str], outputs: List[str]) -> np.ndarray:
		"""Return sentence-level BLEU (0-1) per sample."""
		scores = []
		for ref, hyp in zip(references, outputs):
			ref_text = ref if ref and isinstance(ref, str) else ""
			hyp_text = hyp if hyp and isinstance(hyp, str) else ""
			bleu = sacrebleu.sentence_bleu(hyp_text, [ref_text]).score / 100.0
			scores.append(bleu)
		return np.clip(np.array(scores, dtype=np.float32), 0.0, 1.0)

	def compute_metrics(self, sources: List[str], references: List[str], outputs: List[str]) -> Dict[str, np.ndarray]:
		"""Return STA, SIM, FL, BLEU, and joint J scores per sample."""
		sta = self.compute_sta(outputs)
		fl = self.compute_fl(outputs)
		sim = self.compute_sim(sources, outputs)
		bleu = self.compute_sentence_bleu(references, outputs)
		j_scores = np.clip(sta * sim * fl, 0.0, 1.0)
		return {
			"sta": sta,
			"sim": sim,
			"fl": fl,
			"j": j_scores,
			"bleu": bleu,
		}


def generate_sequences(model, tokenizer, prompts: List[str], device: torch.device,
					   max_input_length: int, gen_kwargs: Dict, show_progress: bool = False,
					   progress_desc: str = "Generating") -> List[str]:
	"""Run batched text generation with optional tqdm progress display."""
	outputs = []
	batch_size = gen_kwargs.get("batch_size", 8)
	use_autocast = gen_kwargs.get("use_autocast", False)
	iterator = range(0, len(prompts), batch_size)
	if show_progress:
		iterator = tqdm(iterator, desc=progress_desc, leave=False)
	for start in iterator:
		batch_prompts = prompts[start:start + batch_size]
		encodings = tokenizer(
			batch_prompts,
			max_length=max_input_length,
			truncation=True,
			padding=True,
			return_tensors="pt",
		).to(device)
		with torch.no_grad():
			with autocast_context(device, use_autocast):
				sequences = model.generate(
				**encodings,
				max_new_tokens=gen_kwargs.get("max_new_tokens", 60),
				num_beams=gen_kwargs.get("num_beams", 5),
				do_sample=gen_kwargs.get("do_sample", False),
				temperature=gen_kwargs.get("temperature", 1.0),
				top_p=gen_kwargs.get("top_p", 1.0),
				no_repeat_ngram_size=gen_kwargs.get("no_repeat_ngram_size", 3),
				pad_token_id=tokenizer.pad_token_id,
			)
		decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
		outputs.extend([item.strip() if item.strip() else " " for item in decoded])
	return outputs


def compute_sequence_logprobs(model, tokenizer, input_ids, attention_mask, response_ids):
	"""Return summed log-probabilities (and optional values) for sequences."""
	decoder_input_ids = response_ids[:, :-1]
	labels = response_ids[:, 1:].clone()
	outputs = model(
		input_ids=input_ids,
		attention_mask=attention_mask,
		decoder_input_ids=decoder_input_ids,
		use_cache=False,
		return_dict=True,
	)
	if hasattr(outputs, "logits"):
		logits = outputs.logits
	else:
		logits = outputs[0]
	log_probs = torch.log_softmax(logits, dim=-1)
	label_mask = (labels != tokenizer.pad_token_id)
	token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
	token_log_probs = token_log_probs * label_mask
	seq_logprob = token_log_probs.sum(dim=-1)

	value_estimate = None
	values = getattr(outputs, "value", None)
	if values is None and isinstance(outputs, tuple) and len(outputs) > 1:
		values = outputs[1]
	if values is not None:
		values = values.squeeze(-1)
		values = values[:, :-1]
		denom = label_mask.sum(dim=-1).clamp(min=1)
		value_estimate = (values * label_mask).sum(dim=-1) / denom
	return seq_logprob, value_estimate


@dataclass
class KLController:
	"""Adaptive KL controller that expands/contracts penalty dynamically."""

	coef: float
	target: float
	rate: float
	min_coef: float = 0.02
	max_coef: float = 1.0

	def update(self, current_kl: float) -> float:
		if abs(current_kl) > self.target * 1.5:
			self.coef = min(self.max_coef, self.coef * (1.0 + self.rate))
		elif abs(current_kl) < self.target * 0.5:
			self.coef = max(self.min_coef, self.coef * (1.0 - self.rate))
		return self.coef


def ppo_single_step(
	policy_model,
	reference_model,
	tokenizer,
	batch,
	reward_scorer,
	metric_evaluator,
	args,
	device,
	optimizer,
	kl_ctrl: KLController,
	step_index: int,
) -> Dict:
	"""Run a single PPO policy/value update and return loss statistics."""

	policy_model.train()
	optimizer.zero_grad()  # clear grads before backward pass

	input_ids = batch["input_ids"].to(device)
	attention_mask = batch["attention_mask"].to(device)
	prompt_texts = batch["prompt_text"]
	reference_texts = batch["reference_text"]

	with torch.no_grad():
		with autocast_context(device, args.fp16_generation):
			response_ids = policy_model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				pad_token_id=tokenizer.pad_token_id,
				max_new_tokens=args.max_new_tokens,
				min_new_tokens=8,
				do_sample=True,
				top_p=0.85,
				temperature=0.7,
				no_repeat_ngram_size=3,
			)

	responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
	rewards = reward_scorer.compute_rewards(
		prompt_texts,
		responses,
		alpha=args.reward_alpha,
		beta=args.reward_beta,
		min_len=args.reward_min_len,
	)
	reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
	metric_stats = {}
	metric_outputs = None
	metric_active = (
		metric_evaluator is not None
		and args.metric_reward_weight > 0.0
		and (step_index % args.metric_reward_stride == 0)
	)
	bleu_active = (
		metric_evaluator is not None
		and args.bleu_reward_weight > 0.0
		and (step_index % args.bleu_reward_stride == 0)
	)
	if metric_active or bleu_active:
		metric_outputs = metric_evaluator.compute_metrics(prompt_texts, reference_texts, responses)

	total_metric_weight = 0.0
	metric_contrib = torch.zeros_like(reward_tensor)
	if metric_active and metric_outputs is not None:
		metric_raw = metric_outputs["j"]
		metric_centered = metric_raw - float(np.mean(metric_raw))
		metric_std = float(np.std(metric_raw) + 1e-6)
		metric_z = np.clip(metric_centered / metric_std, -2.5, 2.5)
		metric_reward = torch.tensor(metric_z, dtype=torch.float32, device=device)
		metric_contrib += args.metric_reward_weight * metric_reward
		total_metric_weight += args.metric_reward_weight
		metric_stats.update({
			"metric_reward_mean": float(metric_reward.mean().item()),
			"metric_reward_std": float(metric_reward.std().item()),
			"metric_j_mean": float(metric_raw.mean()),
			"metric_sta_mean": float(metric_outputs["sta"].mean()),
			"metric_sim_mean": float(metric_outputs["sim"].mean()),
			"metric_fl_mean": float(metric_outputs["fl"].mean()),
		})
	if bleu_active and metric_outputs is not None:
		bleu_raw = metric_outputs["bleu"]
		bleu_centered = bleu_raw - float(np.mean(bleu_raw))
		bleu_std = float(np.std(bleu_raw) + 1e-6)
		bleu_z = np.clip(bleu_centered / bleu_std, -2.5, 2.5)
		bleu_reward = torch.tensor(bleu_z, dtype=torch.float32, device=device)
		metric_contrib += args.bleu_reward_weight * bleu_reward
		total_metric_weight += args.bleu_reward_weight
		metric_stats.update({
			"bleu_reward_mean": float(bleu_reward.mean().item()),
			"bleu_reward_std": float(bleu_reward.std().item()),
			"metric_bleu_mean": float(bleu_raw.mean()),
		})

	if total_metric_weight > 0.0:
		base_weight = max(0.0, 1.0 - total_metric_weight)
		reward_tensor = reward_tensor * base_weight + metric_contrib

	policy_logprob, value_estimate = compute_sequence_logprobs(
		policy_model, tokenizer, input_ids, attention_mask, response_ids
	)
	if value_estimate is None:
		value_estimate = torch.zeros_like(policy_logprob)

	with torch.no_grad():
		ref_logprob, _ = compute_sequence_logprobs(
			reference_model, tokenizer, input_ids, attention_mask, response_ids
		)

	advantages = reward_tensor - value_estimate.detach()
	advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

	ratios = torch.exp(policy_logprob - ref_logprob)
	unclipped = advantages * ratios
	clipped = advantages * torch.clamp(ratios, 1 - args.ppo_clip_range, 1 + args.ppo_clip_range)
	policy_loss = -torch.mean(torch.min(unclipped, clipped))

	value_loss = F.mse_loss(value_estimate, reward_tensor)
	kl_div = torch.mean(ref_logprob - policy_logprob)
	kl_value = kl_div.item()
	kl_coef = kl_ctrl.update(kl_value)
	total_loss = policy_loss + 0.5 * value_loss + kl_coef * torch.abs(kl_div)

	total_loss.backward()  # PPO backward pass
	torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
	optimizer.step()

	return {
		"loss": total_loss.item(),
		"policy_loss": policy_loss.item(),
		"value_loss": value_loss.item(),
		"kl_div": kl_value,
		"reward_mean": float(reward_tensor.mean().item()),
		"kl_coef": kl_coef,
		**{k: metric_stats.get(k) for k in [
			"metric_reward_mean",
			"metric_reward_std",
			"metric_j_mean",
			"metric_sta_mean",
			"metric_sim_mean",
			"metric_fl_mean",
			"bleu_reward_mean",
			"bleu_reward_std",
			"metric_bleu_mean",
		]},
	}


def evaluate_policy(model, tokenizer, prompts: List[str], refs: List[str], reward_scorer,
					args, device) -> Dict:
	"""Generate on held-out prompts and summarize toxicity/similarity/reward."""
	sample_prompts = prompts[:args.eval_sample_size]
	sample_refs = refs[:args.eval_sample_size]
	outputs = generate_sequences(
		model,
		tokenizer,
		sample_prompts,
		device,
		args.max_input_length,
		{
			"batch_size": args.ppo_batch_size,
			"max_new_tokens": args.max_new_tokens,
			"num_beams": 5,
			"no_repeat_ngram_size": 3,
			"use_autocast": args.fp16_generation,
		   },
		   show_progress=False,
	)
	tox = reward_scorer.score_toxicity(outputs)
	sim = reward_scorer.score_similarity(sample_prompts, outputs)
	rewards = reward_scorer.compute_rewards(
		sample_prompts,
		outputs,
		args.reward_alpha,
		args.reward_beta,
		args.reward_min_len,
	)
	preview = pd.DataFrame({
		"prompt": sample_prompts[:5],
		"reference": sample_refs[:5],
		"prediction": outputs[:5],
		"toxicity": tox[:5],  # lightweight qualitative inspection artifact
		"similarity": sim[:5],
		"reward": rewards[:5],
	})
	preview_path = os.path.join(args.output_dir, "eval_preview.csv")
	preview.to_csv(preview_path, index=False)
	return {
		"samples": len(sample_prompts),
		"toxicity_mean": float(np.mean(tox)),
		"toxicity_std": float(np.std(tox)),
		"similarity_mean": float(np.mean(sim)),
		"reward_mean": float(np.mean(rewards)),
		"reward_std": float(np.std(rewards)),
	}


def run_ppo_training(policy_model, reference_model, tokenizer, rl_loader, rl_eval_prompts,
			 rl_eval_refs, reward_scorer, metric_evaluator, args, device) -> Tuple[List[Dict], List[Dict]]:
	"""Execute PPO epochs and collect per-step and per-epoch statistics."""
	optimizer = AdamW(policy_model.parameters(), lr=args.ppo_lr)
	kl_ctrl = KLController(coef=args.ppo_kl_init, target=args.ppo_kl_target, rate=args.ppo_kl_rate)
	training_history = []
	evaluation_history = []

	loader_iter = iter(rl_loader)

	global_step = 0
	for epoch in range(1, args.ppo_epochs + 1):
		print(f"Epoch {epoch}/{args.ppo_epochs}")
		for step in range(1, args.ppo_steps_per_epoch + 1):
			try:
				batch = next(loader_iter)
			except StopIteration:
				loader_iter = iter(rl_loader)
				batch = next(loader_iter)

			stats = ppo_single_step(
				policy_model,
				reference_model,
				tokenizer,
				batch,
				reward_scorer,
				metric_evaluator,
				args,
				device,
				optimizer,
				kl_ctrl,
				global_step + 1,
			)
			global_step += 1
			stats.update({
				"epoch": epoch,
				"step": step,
				"global_step": global_step,
			})
			print(
				f"  Step {step:02d}: reward={stats['reward_mean']:.4f} "
				f"loss={stats['loss']:.4f} kl={stats['kl_div']:.4f}"
			)
			training_history.append(stats)

		eval_metrics = evaluate_policy(
			policy_model,
			tokenizer,
			rl_eval_prompts,
			rl_eval_refs,
			reward_scorer,
			args,
			device,
		)
		eval_metrics.update({"epoch": epoch, "global_step": global_step})
		print(f"  Eval reward={eval_metrics['reward_mean']:.4f} toxicity={eval_metrics['toxicity_mean']:.4f}")
		evaluation_history.append(eval_metrics)

	return training_history, evaluation_history


def save_training_logs(output_dir: str, training_history: List[Dict], evaluation_history: List[Dict]):
	"""Persist PPO training/eval history as CSV for later inspection."""
	if training_history:
		df = pd.DataFrame(training_history)
		df.to_csv(os.path.join(output_dir, "ppo_training_log.csv"), index=False)
	if evaluation_history:
		df = pd.DataFrame(evaluation_history)
		df.to_csv(os.path.join(output_dir, "ppo_eval_log.csv"), index=False)


def run_inference_and_metrics(model_path: str, tokenizer_path: str, tokenizer, device,
							  test_df: pd.DataFrame, output_dir: str, args):
	"""Generate detox outputs, dump TSV, and run ParaDetox metrics."""
	print("Running inference over held-out set...")
	hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
	hf_model.eval()

	dataset = [{"input": row["input"], "target": row["target"]} for _, row in test_df.iterrows()]

	inputs = [item["input"] for item in dataset]
	targets = [item["target"] for item in dataset]

	predictions = generate_sequences(
		hf_model,
		tokenizer,
		inputs,
		device,
		args.max_input_length,
		{
			"batch_size": args.inference_batch_size,
			"max_new_tokens": args.max_new_tokens,
			"num_beams": 5,
			"no_repeat_ngram_size": 3,
			"use_autocast": args.fp16_generation,
		},
		show_progress=True,
		progress_desc="Generating test predictions",
	)

	results_df = pd.DataFrame({
		"toxic": inputs,
		"neutral1": targets,
		"model_output": predictions,
	})
	tsv_path = os.path.join(output_dir, "test_results.tsv")
	results_df.to_csv(tsv_path, sep="\t", index=False)
	print(f"Saved predictions to {tsv_path}")

	print("Computing ParaDetox metrics...")
	metrics = evaluate_paradetox(
		tsv_path=tsv_path,
		output_col="model_output",
		reference_col="neutral1",
		source_col="toxic",
	)
	metrics_path = os.path.join(output_dir, "metrics.txt")
	with open(metrics_path, "w", encoding="utf-8") as f:
		for key, value in metrics.items():
			f.write(f"{key}: {value}\n")
	print(f"Metrics written to {metrics_path}")


def main():
	"""Script entry point covering RLHF training or evaluation-only mode."""
	args = parse_args()
	ensure_dir(args.output_dir)
	final_model_path = os.path.join(args.output_dir, "final_model")

	set_seed(args.seed)
	device = resolve_device()
	print(f"Using device: {device}")

	if args.fp16_generation is None:
		args.fp16_generation = device.type in {"cuda", "mps"}
	args.inference_batch_size = max(1, args.inference_batch_size)

	# Always load and prepare data (needed even for evaluation-only runs)
	raw_df = read_data(args.data_path)
	processed_df = prepare_data(raw_df)
	train_df, val_df, test_df = split_dataset(processed_df, args.seed)

	export_config(args, args.output_dir, {
		"device": str(device),
		"train_size": len(train_df),
		"val_size": len(val_df),
		"test_size": len(test_df),
	})

	if model_exists(final_model_path):
		print(f"Detected existing RLHF model under {final_model_path}. Skipping training.")
		tokenizer = AutoTokenizer.from_pretrained(final_model_path)
		run_inference_and_metrics(
			final_model_path,
			final_model_path,
			tokenizer,
			device,
			test_df,
			args.output_dir,
			args,
		)
		return

	base_model_path = resolve_base_model_path(args.input_dir, args.model_name)
	tokenizer = AutoTokenizer.from_pretrained(base_model_path)

	print(f"Loading policy initialization from {base_model_path}")
	policy_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(base_model_path).to(device)
	reference_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path).to(device)

	rl_train_subset = train_df.sample(
		n=min(args.rl_train_size, len(train_df)),
		random_state=args.seed,
	).reset_index(drop=True)
	rl_eval_subset = val_df.sample(
		n=min(args.rl_eval_size, len(val_df)),
		random_state=args.seed,
	).reset_index(drop=True)

	rl_train_prompts = rl_train_subset["input"].tolist()  # PPO training prompts
	rl_train_refs = rl_train_subset["target"].tolist()
	rl_eval_prompts = rl_eval_subset["input"].tolist()  # held-out prompts for eval
	rl_eval_refs = rl_eval_subset["target"].tolist()

	rl_loader = build_prompt_loader(
		rl_train_prompts,
		rl_train_refs,
		tokenizer,
		args.ppo_batch_size,
		args.max_input_length,
		shuffle=True,
	)

	reward_scorer = RewardScorer(device)  # shared toxicity + similarity scorers
	metric_evaluator = None
	use_metric_eval = (args.metric_reward_weight > 0.0) or (args.bleu_reward_weight > 0.0)
	if use_metric_eval:
		args.metric_reward_weight = float(np.clip(args.metric_reward_weight, 0.0, 1.0))
		args.metric_reward_stride = max(1, args.metric_reward_stride)
		args.bleu_reward_weight = float(np.clip(args.bleu_reward_weight, 0.0, 1.0))
		args.bleu_reward_stride = max(1, args.bleu_reward_stride)
		metric_evaluator = ParaDetoxMiniEvaluator(device)

	training_history, evaluation_history = run_ppo_training(
		policy_model,
		reference_model,
		tokenizer,
		rl_loader,
		rl_eval_prompts,
		rl_eval_refs,
		reward_scorer,
		metric_evaluator,
		args,
		device,
	)

	ensure_dir(final_model_path)
	ppo_policy_path = os.path.join(args.output_dir, "ppo_policy")
	ensure_dir(ppo_policy_path)

	# Persist PPO policy (with value head) for possible continued RL training
	policy_model.save_pretrained(ppo_policy_path)
	tokenizer.save_pretrained(ppo_policy_path)
	print(f"RLHF PPO policy saved to {ppo_policy_path}")

	# Save a plain seq2seq model for inference/metrics to avoid value-head warnings
	policy_model.pretrained_model.save_pretrained(final_model_path)
	tokenizer.save_pretrained(final_model_path)
	print(f"RLHF model saved to {final_model_path}")

	save_training_logs(args.output_dir, training_history, evaluation_history)

	run_inference_and_metrics(
		final_model_path,
		final_model_path,
		tokenizer,
		device,
		test_df,
		args.output_dir,
		args,
	)


if __name__ == "__main__":
	main()
