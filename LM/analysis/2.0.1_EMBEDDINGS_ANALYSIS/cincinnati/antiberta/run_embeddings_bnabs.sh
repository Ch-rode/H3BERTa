#!/bin/bash
#SBATCH --job-name=embeddings_bnabs
#SBATCH --output=embeddings_bnabs.out
#SBATCH --error=embeddings_bnabs.err
#SBATCH --gres=gpu:a100:1

# Esegui lo script python
python 1.run_embeddings_bnabs.py

