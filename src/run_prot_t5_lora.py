"""
run_prot_t5_lora.py
───────────────────
ProtT5-XL encoder  |  LoRA fine-tuning of the pLM backbone

Trainable parameters
  • LoRA adapter weights in the T5 attention Q and V projections
  • All MLP head weights
"""

from fitness_module import run_training

BASE = "/workspace"

run_training(
    # Encoder
    encoder_name = "prot_t5",
    use_lora     = True,
    lora_r       = 8,
    lora_alpha   = 16,
    lora_dropout = 0.05,
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
