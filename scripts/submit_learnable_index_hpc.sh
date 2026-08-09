#!/usr/bin/env bash
#SBATCH --job-name=li_wt1024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=learnable_index_%x_%j.out
#SBATCH --error=learnable_index_%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/learnable_index_wikitext1024_hpc.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi
CONFIG_DIR="$(cd -- "$(dirname -- "${CONFIG_PATH}")" && pwd)"
CONFIG_PATH="${CONFIG_DIR}/$(basename -- "${CONFIG_PATH}")"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-not-set}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "PYTHON_BIN=${PYTHON_BIN}"
nvidia-smi

exec "${PYTHON_BIN}" -u \
  "${PROJECT_ROOT}/scripts/run_learnable_index_hpc.py" \
  --config "${CONFIG_PATH}"
