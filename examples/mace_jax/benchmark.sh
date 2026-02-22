#!/usr/bin/env bash
set -euo pipefail

# Run with a conda env active (so `python` resolves correctly).
# This script benchmarks sync vs async data loading for 5 epochs.
#
# Requirements:
# - Uses one MPI rank per GPU (`mpirun -np <ndev>`).
# - Each rank must see exactly one GPU; we do this via `run_with_rank_gpu.sh`,
#   which sets CUDA_VISIBLE_DEVICES based on the MPI *local* rank.
# - Global batch size (summed over ranks) is identical for all GPU counts.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MPIRUN="${MPIRUN:-mpirun}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-5}"
GLOBAL_BATCH="${GLOBAL_BATCH:-32}"

# MPI CPU binding:
# By default, try to disable binding (Open MPI) so ranks can use all cores.
# Override explicitly if needed, e.g.:
#   MPIRUN_FLAGS="--bind-to none --map-by slot" ./benchmark.sh
#   MPIRUN_FLAGS="" ./benchmark.sh
MPIRUN_FLAGS="${MPIRUN_FLAGS:-}"
MPIRUN_FLAGS_ARR=()
if [[ -n "$MPIRUN_FLAGS" ]]; then
    read -r -a MPIRUN_FLAGS_ARR <<< "$MPIRUN_FLAGS"
else
    if "$MPIRUN" --version 2>/dev/null | grep -qi "Open MPI"; then
        MPIRUN_FLAGS_ARR=(--bind-to none --map-by slot)
    fi
fi

# Provide GPU ids as a comma-separated list.
# Example: GPU_IDS="4,6,7,8" ./benchmark.sh
GPU_IDS="${GPU_IDS:-0}"

IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
if [[ "${#GPU_ARR[@]}" -lt 1 ]]; then
    echo "GPU_IDS must contain at least 1 id (got: '$GPU_IDS')" >&2
    exit 2
fi

max_devices="${#GPU_ARR[@]}"

# Build device counts: 1,2,4,... plus max (if needed), then run from max to fewer.
counts=()
n=1
while (( n <= max_devices )); do
    counts+=("$n")
    n=$((n * 2))
done
if [[ "${counts[$((${#counts[@]} - 1))]}" != "$max_devices" ]]; then
    counts+=("$max_devices")
fi

for ((idx=${#counts[@]}-1; idx>=0; idx--)); do
    ndev="${counts[$idx]}"

    # Visible GPU list for the first ndev GPUs from GPU_IDS.
    vis="${GPU_ARR[0]}"
    for ((i=1; i<ndev; i++)); do
        vis+="${vis:+,}${GPU_ARR[$i]}"
    done

    if (( GLOBAL_BATCH % ndev != 0 )); then
        echo "GLOBAL_BATCH=$GLOBAL_BATCH must be divisible by ndev=$ndev" >&2
        exit 2
    fi
    local_batch=$((GLOBAL_BATCH / ndev))
    if (( local_batch * ndev != GLOBAL_BATCH )); then
        echo "BUG: effective global batch=$((local_batch * ndev)) != GLOBAL_BATCH=$GLOBAL_BATCH" >&2
        exit 2
    fi

    for async in true false; do
        if [[ "$async" == "true" ]]; then
            async_flag="--async_dataloading"
        else
            async_flag="--no-async_dataloading"
        fi

        outdir="output_loader_hdf5_async_${async}_devices_${ndev}"
        if [[ -d "$outdir" ]]; then
            echo "Skipping existing $outdir"
        else
            mkdir -p "$outdir"
            echo "=== loader=hdf5 async=$async devices=$ndev CUDA_VISIBLE_DEVICES=$vis ===" | tee "${outdir}/run.log"
            echo "Launching: $MPIRUN -np $ndev (per-rank batch=$local_batch, global batch=$GLOBAL_BATCH)" | tee -a "${outdir}/run.log"
            /usr/bin/time -p -o "${outdir}/time.txt" \
                "$MPIRUN" "${MPIRUN_FLAGS_ARR[@]}" -np "$ndev" \
                ./run_with_rank_gpu.sh "$vis" \
                "$PYTHON" train_spice_example.py \
                    --epochs "$EPOCHS" \
                    --batch "$local_batch" \
                    $async_flag \
                    --outdir "$outdir" \
                2>&1 | tee -a "${outdir}/run.log"
        fi
    done

    # Numpy loader benchmark: run for all device counts, but only async=false.
    outdir="output_loader_numpy_async_false_devices_${ndev}"
    if [[ -d "$outdir" ]]; then
        echo "Skipping existing $outdir"
    else
        mkdir -p "$outdir"
        echo "=== loader=numpy async=false devices=$ndev CUDA_VISIBLE_DEVICES=$vis ===" | tee "${outdir}/run.log"
        echo "Launching: $MPIRUN -np $ndev (per-rank batch=$local_batch, global batch=$GLOBAL_BATCH)" | tee -a "${outdir}/run.log"
        /usr/bin/time -p -o "${outdir}/time.txt" \
            "$MPIRUN" "${MPIRUN_FLAGS_ARR[@]}" -np "$ndev" \
            ./run_with_rank_gpu.sh "$vis" \
            "$PYTHON" train_spice_example.py \
                --epochs "$EPOCHS" \
                --batch "$local_batch" \
                --no-async_dataloading \
                --numpy_loader \
                --outdir "$outdir" \
            2>&1 | tee -a "${outdir}/run.log"
    fi
done

echo "Done. Check output_loader_*/time.txt for runtimes."