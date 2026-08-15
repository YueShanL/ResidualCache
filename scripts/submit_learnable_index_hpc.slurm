#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Thu Aug 06 2026 12:39:05 GMT+1000 (澳大利亚东部标准时间)

# Partition for the job:
#SBATCH --partition=gpu-l40s
# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1
#SBATCH --job-name=lia-hpc-pipeline
#SBATCH --output=ResidualCache/out/lia-hpc-pipeline-%j.out
#SBATCH --error=ResidualCache/out/lia-hpc-pipeline-%j.err
# The project ID which this job should run under:
#SBATCH --account="punim2072"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=128GB
#SBATCH --time=48:00:00

# Use this email address:
#SBATCH --mail-user=yueshanL1@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END
# check that the script is launched with sbatch

if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

set -euo pipefail

PROJECT_DIR="/data/gpfs/projects/punim2072/LoRA_ICL_Analysis/ResidualCache"

cd "$PROJECT_DIR"

CONFIG_PATH="${1:-${PROJECT_DIR}/configs/learnable_index_wikitext4096_hpc.json}"


if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

source "/data/gpfs/projects/punim2072/LoRA_ICL_Analysis/LLM_LoRA_ICL_Analysis/venv/bin/activate" # using the same venv as icl_analysis

export PIP_CACHE_DIR="/data/gpfs/projects/punim2072/LoRA_ICL_Analysis/LLM_LoRA_ICL_Analysis/.cache/pip"
export HF_HOME="/data/gpfs/projects/punim2072/LoRA_ICL_Analysis/LLM_LoRA_ICL_Analysis/.cache/huggingface"
export HF_HUB_CACHE="/data/gpfs/projects/punim2072/LoRA_ICL_Analysis/LLM_LoRA_ICL_Analysis/.cache/huggingface/hub"
export HF_DATASETS_CACHE="/data/gpfs/projects/punim2072/LoRA_ICL_Analysis/LLM_LoRA_ICL_Analysis/.cache/huggingface/datasets"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME}/token}"
export TORCH_HOME="/data/gpfs/projects/punim2072/LoRA_ICL_Analysis/LLM_LoRA_ICL_Analysis/.cache/torch"

if [[ -z "${HF_TOKEN:-}" && ! -s "${HF_TOKEN_PATH}" ]]; then
  echo "Warning: no Hugging Face token found via HF_TOKEN or HF_TOKEN_PATH=${HF_TOKEN_PATH}" >&2
fi

echo "Project: $PROJECT_DIR"
echo "Python: $(which python)"
python --version
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

python ${PROJECT_DIR}/scripts/run_learnable_index_hpc.py \
  --config "${CONFIG_PATH}"
