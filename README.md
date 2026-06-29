# Rosetta Energy Score Prediction

This folder contains a clean project structure for the bachelor thesis code availability repository.

## Structure

- `src/`: final source code for the protein language model fitness prediction models.
- `notebooks/baselines/`: baseline notebooks for one-hot regression, embedding regression, and MLP embedding models.
- `data/`: dataset documentation and small sequence inputs. Large CSV datasets are not committed.
- `scripts/`: data preparation utilities.
- `plotting/`: plotting script and selected training metrics/figures.
- `cluster/`: SLURM batch scripts used for training runs.

## Main Files

- Final model implementation: `src/fitness_module.py`
- ESM-2 LoRA run script: `src/run_esm2_lora.py`
- ProtT5 LoRA run script: `src/run_prot_t5_lora.py`
- Main baseline dataset, expected locally: `data/normalized_df_split.csv`
- Final model dataset, expected locally: `data/data_split_normalized_combined.csv`

## Baselines

The baseline notebooks are:

- `notebooks/baselines/04_linear_regression_oh.ipynb`
- `notebooks/baselines/05_linear_regression_embeddings.ipynb`
- `notebooks/baselines/06_MLP_embeddings.ipynb`

## Notes

Large datasets and generated artifacts such as model checkpoints, logs, and embedding arrays are not committed to GitHub. If needed, place them in an external archive and document their location here.
