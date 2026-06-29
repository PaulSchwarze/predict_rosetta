
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    SpearmanCorrCoef,
    PearsonCorrCoef,
)

from transformers import (
    AutoModel, AutoTokenizer,
    T5EncoderModel, T5Tokenizer,
    EsmModel, EsmTokenizer,
)
from peft import get_peft_model, LoraConfig


BASE = "/workspace"


# Protein encoders

class ProtT5Encoder(nn.Module):
    MODEL_ID = "Rostlab/prot_t5_xl_half_uniref50-enc"
    LOCAL_DIR = f"{BASE}/prot_t5_local"
    LORA_TARGET_MODULES = ["q", "v"]
    dim = 1024

    def __init__(
        self,
        use_lora: bool,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        src = self.LOCAL_DIR if os.path.exists(self.LOCAL_DIR) else self.MODEL_ID
        print(f"  Loading ProtT5 ({src})  |  LoRA={'ON' if use_lora else 'OFF'} ...")
        self.tokenizer = T5Tokenizer.from_pretrained(src, do_lower_case=False)
        backbone = T5EncoderModel.from_pretrained(src, torch_dtype=torch.float16)

        if use_lora:
            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=self.LORA_TARGET_MODULES,
                bias="none",
            )
            self.backbone = get_peft_model(backbone, cfg)
            self.backbone.print_trainable_parameters()
        else:
            for p in backbone.parameters():
                p.requires_grad = False
            self.backbone = backbone
            total = sum(p.numel() for p in backbone.parameters())
            print(f"  Backbone fully frozen  ({total:,} params, 0 trainable)")

        self.use_lora = use_lora

    def forward(self, sequences: list[str]) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        spaced = [" ".join(list(s)) for s in sequences]

        enc = self.tokenizer(
            spaced,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        ctx = torch.no_grad() if not self.use_lora else torch.enable_grad()
        with ctx:
            hidden = self.backbone(**enc).last_hidden_state

        mask = enc["attention_mask"].unsqueeze(-1).float()
        return (hidden.float() * mask).sum(1) / mask.sum(1)


class ProstT5Encoder(nn.Module):
    MODEL_ID = "Rostlab/ProstT5"
    LOCAL_DIR = f"{BASE}/prost_t5_local"
    LORA_TARGET_MODULES = ["q", "v"]
    dim = 1024

    def __init__(
        self,
        use_lora: bool,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        src = self.LOCAL_DIR if os.path.exists(self.LOCAL_DIR) else self.MODEL_ID
        print(f"  Loading ProstT5 ({src})  |  LoRA={'ON' if use_lora else 'OFF'} ...")
        self.tokenizer = T5Tokenizer.from_pretrained(src, do_lower_case=False)
        backbone = T5EncoderModel.from_pretrained(src, torch_dtype=torch.float16)
        self.dim = getattr(backbone.config, "d_model", self.dim)

        if use_lora:
            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=self.LORA_TARGET_MODULES,
                bias="none",
            )
            self.backbone = get_peft_model(backbone, cfg)
            self.backbone.print_trainable_parameters()
        else:
            for p in backbone.parameters():
                p.requires_grad = False
            self.backbone = backbone
            total = sum(p.numel() for p in backbone.parameters())
            print(f"  Backbone fully frozen  ({total:,} params, 0 trainable)")

        self.use_lora = use_lora

    def forward(self, sequences: list[str]) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        rare_aa_map = str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X"})
        cleaned = [s.upper().translate(rare_aa_map) for s in sequences]
        spaced = ["<AA2fold> " + " ".join(list(s)) for s in cleaned]

        enc = self.tokenizer(
            spaced,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        ctx = torch.no_grad() if not self.use_lora else torch.enable_grad()
        with ctx:
            hidden = self.backbone(**enc).last_hidden_state

        mask = enc["attention_mask"].float()
        mask[:, 0] = 0
        if self.tokenizer.eos_token_id is not None:
            mask = mask * (enc["input_ids"] != self.tokenizer.eos_token_id).float()
        mask = mask.unsqueeze(-1)
        return (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)


class ESM2Encoder(nn.Module):
    MODEL_ID = "facebook/esm2_t33_650M_UR50D"
    LORA_TARGET_MODULES = ["query", "value"]
    dim = 1280

    def __init__(
        self,
        use_lora: bool,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        print(f"  Loading ESM-2 ({self.MODEL_ID})  |  LoRA={'ON' if use_lora else 'OFF'} ...")
        self.tokenizer = EsmTokenizer.from_pretrained(self.MODEL_ID)
        backbone = EsmModel.from_pretrained(self.MODEL_ID)

        if use_lora:
            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=self.LORA_TARGET_MODULES,
                bias="none",
            )
            self.backbone = get_peft_model(backbone, cfg)
            self.backbone.print_trainable_parameters()
        else:
            for p in backbone.parameters():
                p.requires_grad = False
            self.backbone = backbone
            total = sum(p.numel() for p in backbone.parameters())
            print(f"  Backbone fully frozen  ({total:,} params, 0 trainable)")

        self.use_lora = use_lora

    def forward(self, sequences: list[str]) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        enc = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        ctx = torch.no_grad() if not self.use_lora else torch.enable_grad()
        with ctx:
            hidden = self.backbone(**enc).last_hidden_state

        mask = enc["attention_mask"].unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1)


class ESMCEncoder(nn.Module):
    MODEL_ID = "biohub/ESMC-6B"
    LORA_TARGET_MODULES = ["layernorm_qkv.1", "out_proj", "ffn.1", "ffn.3"]
    dim = 960

    def __init__(
        self,
        use_lora: bool,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        print(f"  Loading ESM-C ({self.MODEL_ID})  |  LoRA={'ON' if use_lora else 'OFF'} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        backbone = AutoModel.from_pretrained(self.MODEL_ID, dtype="auto")
        self.dim = getattr(backbone.config, "hidden_size", self.dim)

        if use_lora:
            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=self.LORA_TARGET_MODULES,
                bias="none",
            )
            self.backbone = get_peft_model(backbone, cfg)
            self.backbone.print_trainable_parameters()
        else:
            for p in backbone.parameters():
                p.requires_grad = False
            self.backbone = backbone
            total = sum(p.numel() for p in backbone.parameters())
            print(f"  Backbone fully frozen  ({total:,} params, 0 trainable)")

        self.use_lora = use_lora

    def forward(self, sequences: list[str]) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        enc = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        ctx = torch.no_grad() if not self.use_lora else torch.enable_grad()
        with ctx:
            hidden = self.backbone(**enc).last_hidden_state

        mask = enc["attention_mask"].unsqueeze(-1).float()
        return (hidden.float() * mask).sum(1) / mask.sum(1)


def build_encoder(
    name: str,
    use_lora: bool,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> nn.Module:
    name = name.lower()
    kwargs = dict(
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    if name == "esm2":
        return ESM2Encoder(**kwargs)
    if name in {"esm_c", "esm-c", "esmc"}:
        return ESMCEncoder(**kwargs)
    if name == "prot_t5":
        return ProtT5Encoder(**kwargs)
    if name in {"prost_t5", "prostt5"}:
        return ProstT5Encoder(**kwargs)
    raise ValueError(
        f"Unknown encoder '{name}'. Choose 'esm2', 'esm_c', 'prot_t5', or 'prost_t5'."
    )


# Regression head

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# Lightning model

class FitnessModel(L.LightningModule):
    """
    encode(mutant) âˆ’ encode(wildtype)  â†’  MLP  â†’  scalar fitness score.

    Trainable parameters depend on encoder configuration:
      - use_lora=True  : LoRA adapter weights in the pLM + all MLP head weights
      - use_lora=False : MLP head weights only (pLM backbone fully frozen)
    """

    def __init__(
        self,
        encoder_name: str,
        use_lora: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        hidden_dims: list[int],
        mlp_dropout: float,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = build_encoder(
            encoder_name, use_lora, lora_r, lora_alpha, lora_dropout,
        )
        self.head = MLPHead(self.encoder.dim, hidden_dims, mlp_dropout)
        self.loss_fn = nn.MSELoss()

        self.train_metrics = self._metric_set()
        self.val_metrics = self._metric_set()
        self.test_metrics = self._metric_set()

    @staticmethod
    def _metric_set() -> nn.ModuleDict:
        return nn.ModuleDict({
            "mse":      MeanSquaredError(),
            "mae":      MeanAbsoluteError(),
            "r2":       R2Score(),
            "spearman": SpearmanCorrCoef(),
            "pearson":  PearsonCorrCoef(),
        })

    def forward(self, mut_seqs, wt_seqs):
        return self.head(self.encoder(mut_seqs) - self.encoder(wt_seqs))

    def _shared_step(self, batch, metric_set):
        mut_seqs, wt_seqs, targets = batch
        targets = targets.to(self.device)
        preds = self(mut_seqs, wt_seqs)
        loss = self.loss_fn(preds, targets)

        for m in metric_set.values():
            m(preds, targets)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.train_metrics)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self._log_and_reset(self.train_metrics, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.val_metrics)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self._log_and_reset(self.val_metrics, "val")

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.test_metrics)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        self._log_and_reset(self.test_metrics, "test")

    def _log_and_reset(self, metric_set, prefix):
        for name, metric in metric_set.items():
            self.log(
                f"{prefix}/{name}",
                metric.compute(),
                prog_bar=(name in ("mse", "spearman")),
            )
            metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# Dataset and datamodule

class MutationDataset(Dataset):
    def __init__(self, mut_seqs, wt_seqs, scores) -> None:
        self.mut_seqs = mut_seqs
        self.wt_seqs = wt_seqs
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.mut_seqs[idx], self.wt_seqs[idx], self.scores[idx]


def collate_sequences(batch):
    mut_seqs, wt_seqs, scores = zip(*batch)
    return list(mut_seqs), list(wt_seqs), torch.stack(scores)


class MutationDataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        batch_size: int = 16,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.seed = seed

    def setup(self, stage=None) -> None:
        df = pd.read_csv(self.csv_path)
        total = len(df)

        if self.max_samples is not None and self.max_samples < total:
            split_counts = df["split"].value_counts()
            split_fraction = split_counts / total * self.max_samples
            alloc = split_fraction.apply(np.floor).astype(int)
            remainder = self.max_samples - alloc.sum()
            frac_parts = split_fraction - alloc
            alloc[frac_parts.nlargest(remainder).index] += 1
            rng = np.random.default_rng(self.seed)

            parts = [
                df[df["split"] == s].sample(
                    n=min(n, (df["split"] == s).sum()),
                    random_state=rng.integers(1 << 31),
                )
                for s, n in alloc.items()
            ]
            df = pd.concat(parts).reset_index(drop=True)
            print(f"Subsampled to {len(df):,} / {total:,} rows")
        else:
            print(f"Using full dataset: {total:,} rows")

        counts = df["split"].value_counts().to_dict()
        for s in ["train", "val", "test"]:
            print(f"  {s}: {counts.get(s, 0):,}")

        wt_lookup = dict(zip(df["pdb_fn"], df["Sequence"]))
        self._datasets = {}
        for split in ("train", "val", "test"):
            sub = df[df["split"] == split]
            self._datasets[split] = MutationDataset(
                mut_seqs=sub["mutated Sequence"].tolist(),
                wt_seqs=[wt_lookup[p] for p in sub["pdb_fn"]],
                scores=sub["total_score_z"].values,
            )

    def _loader(self, split, shuffle):
        return DataLoader(
            self._datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_sequences,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def train_dataloader(self): return self._loader("train", True)
    def val_dataloader(self):   return self._loader("val",   False)
    def test_dataloader(self):  return self._loader("test",  False)


# Training function used by the run scripts

def run_training(
    # Encoder
    encoder_name: str,
    use_lora: bool,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # MLP head
    hidden_dims: list[int] = None,
    mlp_dropout: float = 0.3,
    # Optimiser
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    # Data
    csv_path: str = f"{BASE}/data/data_split_normalized_combined.csv",
    batch_size: int = 16,
    max_samples: int | None = None,
    seed: int = 42,
    # Trainer
    epochs: int = 20,
    patience: int = 5,
    log_dir: str = f"{BASE}/logs",
    ckpt_dir: str = f"{BASE}/checkpoints",
    gradient_accumulation_steps: int = 4,
    gradient_clip_val: float = 1.0,
) -> None:
    """
    Build, train, and test a FitnessModel.  All run scripts call this function
    with their specific encoder_name / use_lora combination.
    """
    if hidden_dims is None:
        hidden_dims = [512, 256, 128, 64]

    # Human-readable run tag used for log subdirs and checkpoint filenames
    lora_tag = "lora" if use_lora else "frozen"
    run_name = f"{encoder_name}_{lora_tag}"

    L.seed_everything(seed, workers=True)

    print("=" * 60)
    print(f"  Encoder : {encoder_name.upper()}")
    print(f"  Backbone: {'LoRA (fine-tuned)' if use_lora else 'Frozen (fixed embeddings)'}")
    if use_lora:
        print(f"  LoRA    : r={lora_r}  alpha={lora_alpha}  dropout={lora_dropout}")
    print(f"  Run tag : {run_name}")
    print("=" * 60)

    model = FitnessModel(
        encoder_name=encoder_name,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        hidden_dims=hidden_dims,
        mlp_dropout=mlp_dropout,
        lr=lr,
        weight_decay=weight_decay,
    )

    data = MutationDataModule(
        csv_path=csv_path,
        batch_size=batch_size,
        max_samples=max_samples,
        seed=seed,
    )

    csv_logger = CSVLogger(save_dir=log_dir, name=run_name)
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=run_name)

    callbacks = [
        EarlyStopping(
            monitor="val/loss",
            patience=patience,
            mode="min",
            verbose=True,
            min_delta=0.001,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{run_name}_{{epoch:02d}}_{{val/loss:.4f}}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=10,
        deterministic=False,
        enable_progress_bar=False,
        val_check_interval=0.2,
    )

    trainer.fit(model, datamodule=data)

    ckpt_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")

    trainer.test(model, datamodule=data, ckpt_path="best")

    _plot_from_csv(
        log_dir=csv_logger.log_dir,
        save_dir=csv_logger.log_dir,
        run_name=run_name,
    )


# Plotting

def _plot_from_csv(log_dir: str, save_dir: str, run_name: str = "") -> None:
    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"No metrics file found at {metrics_path} â€” skipping plots.")
        return

    df = pd.read_csv(metrics_path).dropna(subset=["epoch"])
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(run_name, fontsize=14)

    def _plot(ax, col, title):
        if col in df.columns:
            data = df.dropna(subset=[col])
            ax.plot(data["epoch"], data[col], marker="o", markersize=3)
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.3)

    _plot(axes[0, 0], "train/loss_epoch", "Train Loss")
    _plot(axes[0, 1], "val/loss",         "Val Loss (MSE)")
    _plot(axes[0, 2], "val/mse",          "Val MSE")
    _plot(axes[1, 0], "val/spearman",     "Val Spearman Ï")
    _plot(axes[1, 1], "val/pearson",      "Val Pearson r")
    _plot(axes[1, 2], "val/r2",           "Val RÂ²")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {out}")
