#!/bin/bash
#SBATCH --job-name fitness_debug
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir /dss/dsshome1/00/go68gar2/Projects/metl_global
#SBATCH --output /dss/dsshome1/00/go68gar2/Projects/metl_global/logs/slurm_%j_%u_%x.out
#SBATCH --error  /dss/dsshome1/00/go68gar2/Projects/metl_global/logs/slurm_%j_%u_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
# No --gres=gpu line — CPU-only job, gets through the queue immediately

mkdir -p /dss/dsshome1/00/go68gar2/Projects/metl_global/logs

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change RUN_SCRIPT to whichever variant you want to smoke-test.
# MAX_SAMPLES and EPOCHS are overridden via env vars (see fitness_module.py note below).
RUN_SCRIPT="run_prot_t5_lora.py"

srun --container-image /dss/dsshome1/00/go68gar2/nvidia+pytorch+25.04-py3.sqsh \
     --container-mounts /dss/dsshome1/00/go68ар2/Projects/metl_global:/workspace \
     --no-container-mount-home \
     bash -c "
       pip install lightning torchmetrics transformers sentencepiece protobuf peft --quiet &&
       cd /workspace &&
       python - <<'EOF'
# Inline debug runner — overrides MAX_SAMPLES and EPOCHS to keep the job short.
# This imports your real module so the code path is identical to a full run.
import fitness_module as fm

fm.run_training(
    encoder_name = 'prot_t5',   # swap to 'esm2' to test that branch
    use_lora     = True,         # swap to False to test frozen branch
    lora_r       = 8,
    lora_alpha   = 16,
    lora_dropout = 0.05,
    hidden_dims  = [512, 256, 128, 64],
    mlp_dropout  = 0.3,
    lr           = 1e-4,
    weight_decay = 1e-4,
    csv_path     = '/workspace/data/data_split_normalized_combined.csv',
    batch_size   = 4,        # small batch — no GPU memory pressure
    max_samples  = 64,       # 64 rows total: just enough to hit train/val/test
    seed         = 42,
    epochs       = 2,        # 2 epochs is enough to confirm forward+backward pass works
    patience     = 2,
    log_dir      = '/workspace/logs',
    ckpt_dir     = '/workspace/checkpoints',
)
EOF
     "

echo "Job finished at $(date)"
