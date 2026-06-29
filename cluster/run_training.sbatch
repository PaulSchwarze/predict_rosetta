#!/bin/bash
#SBATCH --job-name fitness_lora
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --chdir /home/go68gar2/Projects/metl_global
#SBATCH --output logs/%j/%u_%x.out
#SBATCH --error  logs/%j/%u_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1

# Create log directory for this job
mkdir -p logs/$SLURM_JOB_ID

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

srun --container-image /home/go68gar2/nvidia+pytorch+25.04-py3.sqsh \
     --container-mounts /home/go68gar2/Projects/metl_global:/workspace \
     --no-container-mount-home \
     --pty bash -c "
       pip install lightning torchmetrics pandas matplotlib --quiet &&
       python hpc_test.py
     "

echo "Job finished at $(date)"