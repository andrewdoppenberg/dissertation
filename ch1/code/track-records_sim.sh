#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=24
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --job-name=track-records
#SBATCH --output=track-records.txt

# Submit this script from the directory that contains track-records_sim.py.
cd "$SLURM_SUBMIT_DIR"

module load python-build-bundle/2025a
module load scipy-stack/2025a

# Third-party deps not in scipy-stack. Mesa is pinned to 2.x: the model uses the
# mesa.time / positional-Agent API that Mesa 3.0 removed.
pip install tqdm "mesa>=2,<3"

# Run unbuffered so the tqdm progress bar updates live in the .txt log instead of
# only appearing when the job ends.
export PYTHONUNBUFFERED=1
python -u track-records_sim.py
