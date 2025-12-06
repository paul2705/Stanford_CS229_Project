# Reranking Deliverable

This folder packages the final reranking deliverable requested for the ParaDetox project. The pipeline loads every detox model you specify (covering both SFT baselines under `TRAIN/` and RLHF variants under `RL/`), generates multiple rewrites per toxic sentence, scores each candidate with the lightweight ParaDetox evaluator, and keeps the candidate that maximizes a user-defined objective (e.g., pure J or J + BLEU). The script then exports the reranked outputs and their corpus-level metrics so they can be dropped directly into the final report.

## Quick Start

1. **Activate the existing project environment**
   ```bash
   conda activate bart_detox
   ```
2. **Inspect / customize the config** – `RERANK/configs/example_config.json` already wires up the Bart-base SFT model plus both RLHF variants (J-only & J+BLEU) and the bart-large RLHF checkpoint. Edit the file to:
   - add/remove model entries (any directory that contains a standard Hugging Face `config.json` works),
   - change generation parameters (sampling vs. beam, number of return sequences, max tokens),
   - swap scoring objective weights (e.g., `{"J": 1.0}` for pure J, `{"J": 1.0, "BLEU_NORM": 0.25}` to reward BLEU as well).
3. **Run the pipeline**
   ```bash
   python RERANK/rerank_pipeline.py \
     --config RERANK/configs/example_config.json
   ```
   Use `--override '{"max_samples": 500}'` to temporarily limit dataset size without editing the file.
4. **Full-power run ("满血版")** – To pool *every* SFT + RLHF checkpoint automatically, run:
    ```bash
    python RERANK/build_full_power_config.py \
       --output RERANK/configs/full_power_config.json

    python RERANK/rerank_pipeline.py \
       --config RERANK/configs/full_power_config.json
    ```
    The generated config disables sampling for SFT baselines (clean, deterministic beams) and enables high-temperature sampling on both RLHF families, returning four hypotheses per prompt from every model so the reranker can always pick the globally best rewrite.

Outputs land under the `output_dir` defined in the config (defaults to `RERANK/outputs/example_run`):

| File | Description |
| --- | --- |
| `all_candidates.tsv` | Every generated candidate with its source model、beam/采样序号、完整 ParaDetox 打分（STA/STA_PROB/SIM/FL/FL_PROB/BLEU/BLEU_NORM/J/J_BINARY）以及最终 objective，方便复查。 |
| `reranked_outputs.tsv` | 最终挑选出的回答，每行包含 toxic/参考文本/模型输出、来自哪一个模型 (`chosen_model`)、objective 值，以及该条的 STA/FL/BLEU/J 细分指标（`sta_prob`,`sta`,`sim`,`fl_prob`,`fl`,`bleu`,`j_prob`），满足“结果+来源+指标”一次到位。 |
| `metrics.json` | Corpus-level ParaDetox metrics computed from the reranked outputs (BLEU, STA, SIM, FL, and J). |
| `config.used.json` | Frozen copy of the config after applying any `--override` flags for reproducibility. |

## Configuration Reference

All knobs live in the JSON config; the most relevant sections are:

- `dataset_path`, `source_column`, `reference_column`: point to whichever evaluation split you want to detox. The defaults target `TRAIN/paradetox.tsv` with the `toxic` input column and `neutral1` as the human reference.
- `max_samples`, `sample_strategy`, `sample_seed`: optionally down-sample for quick experiments. `sample_strategy="random"` plus a `sample_seed` will draw a reproducible random subset; omit them to run on the full dataset or simply set `max_samples` + default `head` to grab the first N rows.
- `generation`: global defaults for candidate generation (`max_new_tokens`, `num_beams`, `do_sample`, `top_p`, `temperature`, `num_return_sequences`, `batch_size`, `device`). Each model entry may override these via a nested `generation` block or by lowering the per-model `batch_size` if VRAM is tight.
- `models`: list of `{ "name", "path", "tokenizer", ... }`. Mix and match TRAIN baselines and RLHF checkpoints by pointing to their `final_model` folders. If the tokenizer is stored with the checkpoint, you can omit the `tokenizer` key.
- For the strongest results, use the auto-generated `configs/full_power_config.json` (or re-run the builder after adding new checkpoints). It already wires every TRAIN + RL model with tuned sampling settings and a high-recall objective (`J + 0.25·BLEU + 0.1·SIM + 0.2·FL`).
- 若只想用 RLHF 模型，可直接引用新建的 `configs/rl_only_config.json`（或把 builder 产物裁剪成 RL-only 版本）。该配置默认保留 RL/full_run（J+BLEU）与 RL/full_run_jonly 两套模型，并沿用上面的高召回 objective。
- `scoring`: controls the ParaDetox mini-evaluator. `objective_weights` defines how reranking should behave:
  - Pure J: `{ "J": 1.0 }`
  - J + BLEU: `{ "J": 1.0, "BLEU_NORM": 0.3 }`
  - STA-focused: `{ "STA_PROB": 1.0, "SIM": 0.2 }`, etc.
  You can also move the evaluator to a different device and tweak the toxicity decision threshold.

## How It Works

1. **Candidate generation** – Every listed model receives the toxic prompt. The script can return multiple samples per prompt (controlled by `num_return_sequences` and sampling settings) so the reranker has a diverse pool.
2. **Scoring** – Each candidate is fed through the same lightweight evaluator used during RLHF: Skolkovo toxicity classifier (STA), SentenceTransformer similarity (SIM), CoLA acceptability (FL), plus sentence-level BLEU against the human reference. Both probability-style scores (for smooth objectives) and binary decisions (for official ParaDetox metrics) are recorded.
3. **Reranking** – Candidates are grouped per input sentence, and the one with the highest weighted objective is selected. We keep ties stable in generation order to guarantee determinism.
4. **Evaluation** – The script aggregates STA/SIM/FL exactly like `METRIC/evaluate_paradetox_metrics.py` and runs corpus-level BLEU, reporting the resulting J = STA × SIM × FL. No extra steps are required—you get a ready-to-report metric bundle automatically.

## Extending

- Add more models: simply append another block to `models` (e.g., `TRAIN/mvp/final_model`, `RL/full_run_jonly/bart-large/final_model`).
- Alternative objectives: adjust `objective_weights` or create multiple configs (e.g., `config_j_only.json`, `config_j_bleu.json`) to compare strategies.
- Downstream evaluation: `reranked_outputs.tsv` matches the ParaDetox TSV schema, so you can re-run the official evaluator (`METRIC/evaluate_paradetox_metrics.py --tsv_path ...`) if you want a second opinion.

This setup keeps the original RLHF innovations (mini evaluator, reward mixing) but makes them trivially reusable in a reranking stage, fulfilling the final deliverable requirements.
