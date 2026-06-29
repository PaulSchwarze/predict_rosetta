#!/bin/bash
#SBATCH --job-name fitness_esmc
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --chdir /dss/dsshome1/00/go68gar2/Projects/metl_global
#SBATCH --output /dss/dsshome1/00/go68gar2/Projects/metl_global/logs/slurm_%j_%u_%x.out
#SBATCH --error  /dss/dsshome1/00/go68gar2/Projects/metl_global/logs/slurm_%j_%u_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=lrz-hgx-h100-94x4,mcml-hgx-h100-94x4,lrz-hgx-a100-80x4,mcml-hgx-a100-80x4,lrz-dgx-a100-40x8
#SBATCH --gres=gpu:1

mkdir -p /dss/dsshome1/00/go68gar2/Projects/metl_global/logs

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Select which run script to execute.
# Change this line to switch between lora / frozen variants:
#   run_esmc_lora.py   - LoRA fine-tuning of the backbone
#   run_esmc_frozen.py - frozen backbone, MLP head only
RUN_SCRIPT="run_esmc_lora.py"

srun --container-image /dss/dsshome1/00/go68gar2/nvidia+pytorch+25.04-py3.sqsh \
     --container-mounts /dss/dsshome1/00/go68gar2/Projects/metl_global:/workspace \
     --no-container-mount-home \
     bash -c "
       pip install lightning torchmetrics transformers sentencepiece protobuf peft --quiet &&
       cd /workspace/src &&
       python ${RUN_SCRIPT}
     "

echo "Job finished at $(date)"
