#!/bin/bash
#SBATCH --job-name=ppo_gpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=ppo_logs/ppo_run.out   
#SBATCH --error=ppo_logs/ppo_run.err    

ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
source ~/.bashrc
conda activate grpo

cd /scratch/user/mincheolseong/GRPO_project/ECEN743-GRPO-Project-Proposal/mincheol_runs
python run_ppo.py

