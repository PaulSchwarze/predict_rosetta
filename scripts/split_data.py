import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\paulb\PycharmProjects\METL_GLOBAL\data\normalized_df.csv")

unique_pdbs = df["pdb_fn"].unique()
rng = np.random.default_rng(seed=42)
rng.shuffle(unique_pdbs)

n = len(unique_pdbs)
n_train = round(n * 0.8)
n_val   = round(n * 0.1)

train_pdbs = set(unique_pdbs[:n_train])
val_pdbs   = set(unique_pdbs[n_train : n_train + n_val])
test_pdbs  = set(unique_pdbs[n_train + n_val :])

def assign_split(pdb):
    if pdb in train_pdbs:
        return "train"
    elif pdb in val_pdbs:
        return "val"
    else:
        return "test"

df["split"] = df["pdb_fn"].map(assign_split)

print(f"Train PDFs: {len(train_pdbs)}, Val PDFs: {len(val_pdbs)}, Test PDFs: {len(test_pdbs)}")
print(df["split"].value_counts())

df.to_csv(r"C:\Users\paulb\PycharmProjects\METL_GLOBAL\data\normalized_df_split.csv", index=False)
print("Saved to data/normalized_df_split.csv")