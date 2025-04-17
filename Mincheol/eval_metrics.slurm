#!/bin/bash
#SBATCH --job-name=eval_metrics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=metrics_results/eval_metrics.out
#SBATCH --error=metrics_results/eval_metrics.err

ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
source ~/.bashrc
conda activate grpo

cd /scratch/user/mincheolseong/GRPO_project/ECEN743-GRPO-Project-Proposal/mincheol_runs
python eval_metrics.py

