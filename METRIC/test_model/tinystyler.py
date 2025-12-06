import torch
import importlib
from huggingface_hub import hf_hub_download
from transformers import set_seed
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

df = pd.read_csv("paradetox.tsv", sep="\t")

# Import TinyStyler
tinystyler_module = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "tinystyler",
        hf_hub_download(repo_id="tinystyler/tinystyler", filename="tinystyler.py"),
    )
)
tinystyler_module.__spec__.loader.exec_module(tinystyler_module)
get_tinystyler_model, get_target_style_embeddings = tinystyler_module.get_tinystyler_model, tinystyler_module.get_target_style_embeddings
run_tinystyler_batch = tinystyler_module.run_tinystyler_batch

# Load the TinyStyler model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, model = get_tinystyler_model(device)
tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large")
set_seed(42)

# Define inputs
target_file_path = "neutral.txt"
lines = []
with open(target_file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()       
        if line:               
            lines.append(line)

df["model_output"] = ""

output_buffer = []
flush_interval = 10
row_index = 0

input_file_path = "toxic.txt"
with open(input_file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()     
        if line: 
            source_text = line
            target_texts = lines[10:20]

            # Run TinyStyler
            inputs = tokenizer(
                [source_text], padding="longest", truncation=True, return_tensors="pt"
            ).to(device)
            output = model.generate(
                **inputs,
                style=get_target_style_embeddings([target_texts], device).to(device),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=128,
            )
            generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            
            print(row_index, generated_text) 
            df.at[row_index, "model_output"] = generated_text
            row_index += 1

            # Buffer & flush
            output_buffer.append(generated_text)

            if len(output_buffer) >= flush_interval:
                df.to_csv("paradetox_with_preds.tsv", sep="\t", index=False)
                print(f"Flushed {len(output_buffer)} lines to TSV...")
                output_buffer = []  

df.to_csv("paradetox_with_preds.tsv", sep="\t", index=False)
print("Final flush complete. Saved:", "paradetox_with_preds.tsv")
