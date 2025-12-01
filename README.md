# Stanford_CS229_Project

## Train Models

This training script fine-tunes a sequence-to-sequence model (BART, T5, etc.) for toxic-text detoxification using data formatted in the ParaDetox style. The script performs efficient data preprocessing, model training, prediction generation, and evaluation.

To train the model, follow the steps below in the project root folder:

### Installation
1. Create the Conda Environment:
```
conda env create -f env.yml
```
2. Activate the Environment:
```
conda activate bart_detox
```

### Train Dataset Format (TSV)
Ensure the input file for training (default: /TRAIN/paradetox.tsv) is a Tab-Separated Values (TSV) file with the following columns: ``toxic, neutral1, neutral2, neutral3``

**Note:**

1. Contents in ``neutral2`` and ``neutral3`` are optional but recommended for data augmentation.

2. Short/empty examples are automatically filtered.

### Usage

#### Basic Run
```
python TRAIN/train_detox.py
```

#### Custom Parameters
```
python TRAIN/train_detox.py \
    --model_name MODEL_TO_TRAIN \
    --data_path TRAIN_DATA \
    --output_dir OUTPUT_DIR \
    --epochs 5 \
    --batch_size 8 \
    --lr 5e-5 \
    --max_len 128 \
    --beams 5
```

#### Arguments

| Argument       | Default                 | Description                                                           |
| -------------- | ----------------------- | --------------------------------------------------------------------- |
| `--model_name` | `facebook/bart-base`    | HuggingFace model checkpoint (e.g., `facebook/bart-large`, `t5-base`) |
| `--data_path`  | `./TRAIN/paradetox.tsv` | Path to the ParaDetox-formatted TSV dataset                           |
| `--output_dir` | `./detox_model_output`  | Directory to save trained models, logs, and evaluation results        |
| `--epochs`     | `10`                    | Number of training epochs                                             |
| `--batch_size` | `4`                     | Training and evaluation batch size                                    |
| `--lr`         | `3e-5`                  | Learning rate                                                         |
| `--max_len`    | `128`                   | Maximum token length for inputs and outputs                           |
| `--seed`       | `42`                    | Random seed for reproducibility                                       |
| `--beams`      | `5`                     | Number of beams for generation during decoding                        |

After training, the script automatically runs evaluation metrics (see below for evaluation details).

### Outputs

Trained Model: Saved to ``[OUTPUT_DIR]/final_model``

Predictions: Saved to ``[OUTPUT_DIR]/test_results.tsv``

Metrics: Saved to ``[OUTPUT_DIR]/metrics.txt``

### Training Results

The evaluation metrics for all the models we tested are saved in ``TRAIN/summary_metrics.csv`` and sorted by ``J``.

## Evaluate Metrics

python version: Python 3.13.5

**Use `my_conda_list.txt` to setup conda virtual environment for python3 requirement to run this metric python code OR Use `environment.yml` to create environment directly (but not tested, maybe wrong)**

Example Usage for ``evaluate_paradetox_metrics.py``

```
python3 evaluate_paradetox_metrics.py \                                
  --tsv_path paradetox_with_preds.tsv \
  --output_col model_output \
  --reference_col neutral1 \
  --source_col toxic
```

