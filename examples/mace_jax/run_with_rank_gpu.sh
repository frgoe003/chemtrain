#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_with_rank_gpu.sh "GPU_IDS" <command> [args...]
#
# Example:
#   ./run_with_rank_gpu.sh "5,7" python train_spice_example.py --epochs 5
#
# Picks exactly one GPU per MPI local rank by setting CUDA_VISIBLE_DEVICES.

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"GPU_IDS\" <command> [args...]" >&2
  exit 2
fi

GPU_IDS="$1"
shift

IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
if [[ ${#GPU_ARR[@]} -lt 1 ]]; then
  echo "GPU_IDS must contain at least 1 id (got: '$GPU_IDS')" >&2
  exit 2
fi

# Determine local rank (preferred), with fallbacks across MPI implementations.
LOCAL_RANK=""
if [[ -n "${OMPI_COMM_WORLD_LOCAL_RANK:-}" ]]; then
  LOCAL_RANK="$OMPI_COMM_WORLD_LOCAL_RANK"
elif [[ -n "${MPI_LOCALRANKID:-}" ]]; then
  LOCAL_RANK="$MPI_LOCALRANKID"
elif [[ -n "${SLURM_LOCALID:-}" ]]; then
  LOCAL_RANK="$SLURM_LOCALID"
elif [[ -n "${PMI_LOCAL_RANK:-}" ]]; then
  LOCAL_RANK="$PMI_LOCAL_RANK"
elif [[ -n "${PALS_LOCAL_RANKID:-}" ]]; then
  LOCAL_RANK="$PALS_LOCAL_RANKID"
fi

if [[ -z "$LOCAL_RANK" ]]; then
  echo "Could not determine MPI local rank. Tried OMPI_COMM_WORLD_LOCAL_RANK, MPI_LOCALRANKID, SLURM_LOCALID, PMI_LOCAL_RANK, PALS_LOCAL_RANKID." >&2
  exit 2
fi

if ! [[ "$LOCAL_RANK" =~ ^[0-9]+$ ]]; then
  echo "Invalid local rank: '$LOCAL_RANK'" >&2
  exit 2
fi

if [[ "$LOCAL_RANK" -ge "${#GPU_ARR[@]}" ]]; then
  echo "Local rank $LOCAL_RANK exceeds GPU_IDS list length ${#GPU_ARR[@]} (GPU_IDS='$GPU_IDS')" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ARR[$LOCAL_RANK]}"
export PYTHONUNBUFFERED=1

exec "$@"
