import argparse
import pandas as pd
import numpy as np
import torch
import os
import warnings
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from METRIC.evaluate_paradetox_metrics import *

try:
    # import evaluate
    import sacrebleu
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification
except ImportError:
    print("Warning: Evaluation metrics may be limited.")

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model selection
    parser.add_argument("--model_name", type=str, default="facebook/bart-base", 
                        help="HuggingFace model checkpoint (e.g., facebook/bart-large, t5-base)")
    
    # Data paths
    parser.add_argument("--data_path", type=str, default="./TRAIN/paradetox.tsv", help="Path to the input TSV file")
    parser.add_argument("--output_dir", type=str, default="./TRAIN/detox_model_output", help="Directory to save the model")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and eval")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--beams", type=int, default=5, help="Number of beams for generation")
    
    return parser.parse_args()

# --- Data Preprocessing ---
def prepare_data(df):
    """
    Optimized data preparation:
    1. Uses all neutral versions for data augmentation
    2. Filters short/empty samples
    """
    data_pairs = []
    
    for _, row in df.iterrows():
        toxic_text = str(row['toxic']).strip()
        
        # Skip empty or very short texts
        if not toxic_text or len(toxic_text) < 5:
            continue
        
        # Collect all non-empty neutral versions
        neutral_versions = []
        for col in ['neutral1', 'neutral2', 'neutral3']:
            if col in df.columns and pd.notna(row[col]):
                neutral = str(row[col]).strip()
                if neutral and len(neutral) >= 5:
                    neutral_versions.append(neutral)
        
        # Create pairs for all versions (Data Augmentation)
        for neutral in neutral_versions:
            data_pairs.append({
                'input': toxic_text,
                'target': neutral
            })
    
    return pd.DataFrame(data_pairs)

# --- Main Training Logic ---
def main():
    args = parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Load and Process Data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at {args.data_path}")
        
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path, sep='\t')
    
    processed_df = prepare_data(df)
    print(f"Original data: {len(df)} rows")
    print(f"Processed (augmented) data: {len(processed_df)} rows")
    
    # Split Data
    train_df, temp_df = train_test_split(processed_df, test_size=0.2, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)
    
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df[['input', 'target']].reset_index(drop=True)),
        'validation': Dataset.from_pandas(val_df[['input', 'target']].reset_index(drop=True)),
        'test': Dataset.from_pandas(test_df[['input', 'target']].reset_index(drop=True))
    })

    # 2. Load Model & Tokenizer
    print(f"Loading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = model.to(device)

    # 3. Tokenization
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples['input'],
            max_length=args.max_len,
            truncation=True,
            padding=False
        )
        labels = tokenizer(
            examples['target'],
            max_length=args.max_len,
            truncation=True,
            padding=False
        )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=['input', 'target'],
        desc="Tokenizing"
    )

    # 4. Training Configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_len,
        generation_num_beams=args.beams,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        warmup_steps=500,
        lr_scheduler_type="linear",
        push_to_hub=False,
        report_to="none"
    )

    print(f"\nModel training parameters:")
    print(f"  - Model: {args.model_name} ({model.num_parameters() / 1e6:.2f}M parameters)")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Training epochs: {training_args.num_train_epochs}")
    print(f"  - Warmup steps: {training_args.warmup_steps}")
    print(f"  - Mixed precision: {training_args.fp16}")
    print(f"  - Beam search: {training_args.generation_num_beams} beams")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 5. Training
    print("\nStarting Training...")
    trainer.train()
    
    save_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    print("\nTrained model saved to: {save_model_path}")

    # 6. Evaluation on Test Set
    print("\nGenerating predictions on Test Set...")
    
    # Generate predictions
    predictions = []
    
    test_loader = torch.utils.data.DataLoader(
        tokenized_datasets['test'].remove_columns(['labels']), 
        batch_size=args.batch_size * 2,
        collate_fn=data_collator
    )
    
    raw_test_inputs = test_df['input'].tolist()
    raw_test_targets = test_df['target'].tolist()
    
    model.eval()
    for _, batch in enumerate(tqdm(test_loader, desc="Generating")):
        batch = {k: v.to(device) for k, v in batch.items() if v is not None}
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_length=args.max_len,
                num_beams=args.beams,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_preds = [pred if pred.strip() else " " for pred in decoded_preds]
        predictions.extend(decoded_preds)

    # Save Results to CSV
    results_df = pd.DataFrame({
        "toxic": raw_test_inputs,
        "neutral1": raw_test_targets,
        "model_output": predictions
    })
    tsv_path = os.path.join(args.output_dir, "test_results.tsv")
    results_df.to_csv(tsv_path, sep='\t', index=False)
    print(f"\nDetailed predictions saved to: {tsv_path}")

    # Compute & Save Metrics
    print("\nComputing Metrics...")
    
    try:
        metrics = evaluate_paradetox(
            tsv_path=tsv_path,
            output_col="model_output",
            reference_col="neutral1",
            source_col="toxic"
        )

        metrics_path = os.path.join(args.output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"Metrics saved to {metrics_path}")
    except Exception as e:
        print(f"ERROR: {e}. Evaluation failed.")

if __name__ == "__main__":
    main()