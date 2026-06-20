#!/bin/bash
#SBATCH --job-name=audiobook_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/audiobook_test_%j.out
#SBATCH --error=logs/audiobook_test_%j.err

# Test: synthesize first 10 entries of Fourth Wing Chapter One with XTTS v2
# Run from: /srv/eeg_reconstruction/ugo/audio-tools

mkdir -p logs

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Starting synthesis test..."

cd /srv/eeg_reconstruction/ugo/audio-tools

python test_synthesize_chapter.py
