# Stanford_CS229_Project



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

