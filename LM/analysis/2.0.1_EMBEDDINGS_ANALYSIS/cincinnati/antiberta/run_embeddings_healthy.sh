#!/bin/bash
#SBATCH --job-name=embeddings_healthy
#SBATCH --output=embeddings_healthy.out
#SBATCH --error=embeddings_healthy.err
#SBATCH --gres=gpu:a100:1

# Esegui lo script python
python 1.run_embeddings_healthy.py

