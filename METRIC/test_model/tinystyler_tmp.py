import torch
import importlib
from huggingface_hub import hf_hub_download
from transformers import set_seed
import pandas as pd
from tqdm import tqdm

# =============== 1. Load TinyStyler ===============
# (same as your original code)

tinystyler_module = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "tinystyler",
        hf_hub_download(repo_id="tinystyler/tinystyler", filename="tinystyler.py"),
    )
)
tinystyler_module.__spec__.loader.exec_module(tinystyler_module)
get_tinystyler_model = tinystyler_module.get_tinystyler_model
get_target_style_embeddings = tinystyler_module.get_target_style_embeddings

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, model = get_tinystyler_model(device)
set_seed(42)

# =============== 2. Configs ===============

# Path to your ParaDetox-style TSV
INPUT_TSV = "paradetox.tsv"           # <-- change if needed
OUTPUT_TSV = "paradetox_with_preds.tsv"

SOURCE_COL = "toxic"                  # toxic input
REF_COL = "neutral1"                  # human reference (already there)
OUTPUT_COL = "model_output"           # what we’ll generate

# Target style examples: you can change these if you want another style
target_texts = ["idk.....but i have faith in you lol"]

# Generation settings (you can tweak)
GEN_KWARGS = dict(
    do_sample=True,
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=128,
)


# =============== 3. Helper: batched generation ===============

def generate_style_transfer_batch(text_batch, style_emb):
    """
    text_batch: list[str]
    style_emb : tensor of shape [hidden_dim] or [1, hidden_dim]
    returns: list[str] – generated texts
    """
    # Expand style embedding to match batch size
    if style_emb.dim() == 1:
        style_emb_batch = style_emb.unsqueeze(0).expand(len(text_batch), -1)
    elif style_emb.size(0) == 1:
        style_emb_batch = style_emb.expand(len(text_batch), -1)
    else:
        # assume already correct size
        style_emb_batch = style_emb

    inputs = tokenizer(
        text_batch,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            style=style_emb_batch.to(device),
            **GEN_KWARGS,
        )

    generated = tokenizer.batch_decode(output, skip_special_tokens=True)
    return generated


# =============== 4. Main: read TSV, run model, write TSV ===============

def main():
    print(f"Loading TSV from: {INPUT_TSV}")
    df = pd.read_csv(INPUT_TSV, sep="\t")

    if SOURCE_COL not in df.columns:
        raise ValueError(f"Column '{SOURCE_COL}' not found in {INPUT_TSV}")

    if REF_COL not in df.columns:
        print(f"WARNING: reference column '{REF_COL}' not found; "
              f"BLEU won’t work unless you add it later.")

    toxic_texts = df[SOURCE_COL].astype(str).tolist()
    print(f"Total rows to process: {len(toxic_texts)}")

    # Compute target style embedding ONCE
    print("Computing target style embeddings...")
    with torch.no_grad():
        style_emb = get_target_style_embeddings([target_texts], device)
        # style_emb shape is usually [1, hidden_dim]

    preds = []
    batch_size = 8  # adjust if GPU/CPU memory is tight

    print("Running TinyStyler on toxic sentences...")
    for i in tqdm(range(0, len(toxic_texts), batch_size)):
        batch = toxic_texts[i:i + batch_size]
        gen_batch = generate_style_transfer_batch(batch, style_emb)
        preds.extend(gen_batch)

    assert len(preds) == len(df)

    # Add prediction column
    df[OUTPUT_COL] = preds

    print(f"Writing output TSV to: {OUTPUT_TSV}")
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print("Done! TSV with columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()

