# Data

Expected local datasets:

- `normalized_df_split.csv`: main split dataset used for the baseline experiments.
- `data_split_normalized_combined.csv`: dataset used by the final model training scripts.
- `wildtype_sequences.fasta`: unique wildtype sequences with `pdb_fn` identifiers. This small FASTA file is committed.

The CSV datasets are intentionally not committed because they are too large for the GitHub repository. Place them in this directory locally when running the notebooks or training scripts.

Some baseline notebooks reference precomputed embedding arrays:

- `prot_t5_embeddings_combined.npy`
- `prot_t5_indices_combined.npy`

These are generated artifacts and are not copied here by default because they are large. Add them to `data/` if you want the embedding notebooks to run without regenerating embeddings.
