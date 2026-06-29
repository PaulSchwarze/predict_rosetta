"""
run_prostt5_frozen.py
---------------------
ProstT5 encoder  |  Backbone fully frozen - MLP head only

Trainable parameters
  - MLP head weights only
  - ProstT5 backbone produces fixed embeddings
"""

from fitness_module import run_training

BASE = "/workspace"

run_training(
    # Encoder
    encoder_name = "prostt5",
    use_lora     = False,
    # MLP head
    hidden_dims  = [512, 256, 128, 64],
    mlp_dropout  = 0.3,
    # Optimiser
    lr           = 1e-4,
    weight_decay = 1e-4,
    # Data
    csv_path     = f"{BASE}/data/data_split_normalized_combined.csv",
    batch_size   = 16,
    max_samples  = None,
    seed         = 42,
    # Trainer
    epochs       = 20,
    patience     = 5,
    log_dir      = f"{BASE}/logs",
    ckpt_dir     = f"{BASE}/checkpoints",
)
