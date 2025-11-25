import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import sacrebleu

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sentence_transformers import SentenceTransformer


# -----------------------------
# 0. Utility: batching
# -----------------------------
def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


# -----------------------------
# 1. BLEU
# -----------------------------
def compute_bleu(references, hypotheses):
    """
    references: list[str]  – human detoxified sentences
    hypotheses: list[str]  – model outputs

    Returns BLEU as a percentage (0–100), same scale as sacrebleu.
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score  # paper reports BLEU in 0–100


# -----------------------------
# 2. Style Accuracy (STA)
#    – % of outputs predicted NON-toxic
# -----------------------------
def compute_style_accuracy(
    sentences,
    tox_tokenizer,
    tox_model,
    device="cpu",
    threshold=0.5,
):
    """
    Uses SkolkovoInstitute/roberta_toxicity_classifier
    (same as in the ParaDetox README).

    STA = percentage of sentences classified as NON-toxic.
    """
    tox_model.eval()
    tox_model.to(device)

    probs_toxic = []

    with torch.no_grad():
        for batch in batch_iter(sentences, 32):
            enc = tox_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            out = tox_model(**enc)
            logits = out.logits  # [B, 2]
            # assume label 1 = toxic, 0 = non-toxic
            p = torch.softmax(logits, dim=-1)[:, 1]
            probs_toxic.extend(p.cpu().numpy())

    probs_toxic = np.array(probs_toxic)
    # non-toxic if p(toxic) < threshold
    non_toxic_mask = probs_toxic < threshold
    sta = non_toxic_mask.mean()
    return sta


# -----------------------------
# 3. Content Preservation (SIM)
#    – cosine similarity of paraphrastic embeddings
# -----------------------------
def compute_similarity(
    sources,
    outputs,
    sim_model,
):
    """
    Uses Wieting-style paraphrastic embeddings via
    a sentence-transformers model (e.g. jwieting/paraphrastic_test).

    SIM = mean cosine similarity(source_emb, output_emb).
    """
    # encode with normalization -> cosine is just dot product
    emb_src = sim_model.encode(
        sources,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    emb_out = sim_model.encode(
        outputs,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # cosine similarity per pair
    sims = np.sum(emb_src * emb_out, axis=1)
    return float(sims.mean())


# -----------------------------
# 4. Fluency (FL)
#    – % of outputs judged acceptable by CoLA classifier
# -----------------------------
def compute_fluency(
    sentences,
    cola_tokenizer,
    cola_model,
    device="cpu",
):
    """
    Uses textattack/roberta-base-CoLA (RoBERTa model
    fine-tuned on CoLA acceptability).

    FL = % sentences predicted as 'acceptable' (label 1).
    """
    cola_model.eval()
    cola_model.to(device)

    preds = []

    with torch.no_grad():
        for batch in batch_iter(sentences, 32):
            enc = cola_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            out = cola_model(**enc)
            logits = out.logits  # [B, 2]
            labels = torch.argmax(logits, dim=-1)
            preds.extend(labels.cpu().numpy())

    preds = np.array(preds)
    # In textattack/roberta-base-CoLA, label 1 = acceptable
    acceptable_mask = preds == 1
    fl = acceptable_mask.mean()
    return fl


# -----------------------------
# 5. Joint metric J
# -----------------------------
def compute_joint(sta, sim, fl):
    """
    J = STA * SIM * FL
    """
    return float(sta * sim * fl)


# -----------------------------
# 6. Putting it all together
# -----------------------------
def evaluate_paradetox(
    tsv_path,
    output_col="model_output",
    reference_col="neutral1",
    source_col="toxic",
):
    df = pd.read_csv(tsv_path, sep="\t")

    # Filter rows where we have all three texts
    df = df.dropna(subset=[source_col, reference_col, output_col])

    sources = df[source_col].tolist()
    references = df[reference_col].tolist()
    outputs = df[output_col].tolist()

    print(f"Loaded {len(df)} examples with model outputs.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 6.1 Load models (all public)
    # Toxicity classifier – same as ParaDetox paper
    TOX_MODEL_NAME = "SkolkovoInstitute/roberta_toxicity_classifier"
    tox_tokenizer = AutoTokenizer.from_pretrained(TOX_MODEL_NAME)
    tox_model = AutoModelForSequenceClassification.from_pretrained(TOX_MODEL_NAME)

    # CoLA acceptability classifier
    COLA_MODEL_NAME = "textattack/roberta-base-CoLA"
    cola_tokenizer = AutoTokenizer.from_pretrained(COLA_MODEL_NAME)
    cola_model = AutoModelForSequenceClassification.from_pretrained(COLA_MODEL_NAME)

    # Wieting-style paraphrastic sentence embeddings
    # (any strong paraphrastic model works; this one is close in spirit)
    # SIM_MODEL_NAME = "jwieting/paraphrastic_test"
    # sim_model = SentenceTransformer(SIM_MODEL_NAME)
    SIM_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    sim_model = SentenceTransformer(SIM_MODEL_NAME)

    # 6.2 Compute metrics
    print("\nComputing BLEU...")
    bleu = compute_bleu(references, outputs)  # 0–100

    print("Computing Style Accuracy (STA)...")
    sta = compute_style_accuracy(outputs, tox_tokenizer, tox_model, device=device)

    print("Computing Content Preservation (SIM)...")
    sim = compute_similarity(sources, outputs, sim_model)

    print("Computing Fluency (FL)...")
    fl = compute_fluency(outputs, cola_tokenizer, cola_model, device=device)

    j = compute_joint(sta, sim, fl)

    # 6.3 Print results in same style as paper
    print("\n=== ParaDetox-style Metrics ===")
    print(f"BLEU: {bleu:6.2f}")            # paper reports BLEU in 0–100
    print(f"STA : {sta:6.3f}")
    print(f"SIM : {sim:6.3f}")
    print(f"FL  : {fl:6.3f}")
    print(f"J   : {j:6.3f}")

    return {
        "BLEU": bleu,
        "STA": sta,
        "SIM": sim,
        "FL": fl,
        "J": j,
    }


# -----------------------------
# 7. CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv_path",
        type=str,
        required=True,
        help="Path to paradetox-like TSV file.",
    )
    parser.add_argument(
        "--output_col",
        type=str,
        default="model_output",
        help="Column name with model outputs.",
    )
    parser.add_argument(
        "--reference_col",
        type=str,
        default="neutral1",
        help="Column name with human detoxified reference.",
    )
    parser.add_argument(
        "--source_col",
        type=str,
        default="toxic",
        help="Column name with toxic source sentences.",
    )
    args = parser.parse_args()

    evaluate_paradetox(
        tsv_path=args.tsv_path,
        output_col=args.output_col,
        reference_col=args.reference_col,
        source_col=args.source_col,
    )

