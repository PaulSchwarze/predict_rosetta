#!/bin/bash
#SBATCH --job-name fitness_lora
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --chdir /home/go68gar2/Projects/metl_global
#SBATCH --output /home/go68gar2/Projects/metl_global/logs/slurm_%j_%u_%x.out
#SBATCH --error  /home/go68gar2/Projects/metl_global/logs/slurm_%j_%u_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1

# Slurm writes logs here — must exist before the job starts
mkdir -p /home/go68gar2/Projects/metl_global/logs

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

srun --container-image /home/go68gar2/nvidia+pytorch+25.04-py3.sqsh \
     --container-mounts /home/go68gar2/Projects/metl_global:/workspace \
     --no-container-mount-home \
     bash -c "
       pip install lightning torchmetrics transformers sentencpiece tiktoken peft --quiet &&
       cd /workspace &&
       python run_training_prot_t5_lora.py
     "

echo "Job finished at $(date)"
