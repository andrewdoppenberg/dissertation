#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=24
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --job-name=soul-making
#SBATCH --output=soul-making.txt

# Submit this script from the directory that contains soul-making_sim.py.
cd "$SLURM_SUBMIT_DIR"

module load python-build-bundle/2025a
module load scipy-stack/2025a

# Third-party dep not in scipy-stack.
pip install tqdm

# Run unbuffered so the tqdm progress bar updates live in the .txt log instead of
# only appearing when the job ends. The script reads SLURM's CPU allocation
# itself (resolve_workers), so no --workers argument is needed here.
export PYTHONUNBUFFERED=1
python -u soul-making_sim.py
