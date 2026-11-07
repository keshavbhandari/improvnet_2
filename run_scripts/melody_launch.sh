#!/bin/bash -x
#SBATCH --job-name=improvnet-melody
#SBATCH --account=e-dev-2026d09-047
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/e/home/jusers/bhandari1/jupiter/improvnet_melody_contrastive/run_scripts/melody_logs/slurm-%j.out
#SBATCH --error=/e/home/jusers/bhandari1/jupiter/improvnet_melody_contrastive/run_scripts/melody_logs/slurm-%j.err

source /e/project1/e-dev-2026d09-047/software/miniforge3/etc/profile.d/conda.sh
conda activate improvnet

cd /e/home/jusers/bhandari1/jupiter/improvnet_melody_contrastive || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
# Set to 0 in the submission environment to disable automatic continuation.
export AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"

if command -v git >/dev/null 2>&1; then
    echo "Git branch: $(git branch --show-current)"
    echo "Git commit: $(git rev-parse HEAD)"
    git status --short
fi

CHECKPOINT_REQUEST_FILE="/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v8/.checkpoint-request-${SLURM_JOB_ID}"
export CHECKPOINT_REQUEST_FILE
mkdir -p "$(dirname "$CHECKPOINT_REQUEST_FILE")"
rm -f "$CHECKPOINT_REQUEST_FILE"

TRAINING_STEP_PID=""
submit_successor_after_checkpoint() {
    trap - USR1
    echo "Slurm time limit is approaching; requesting a synchronized checkpoint."
    touch "$CHECKPOINT_REQUEST_FILE"
    if [[ -n "$TRAINING_STEP_PID" ]]; then
        wait "$TRAINING_STEP_PID" || exit $?
    fi
    rm -f "$CHECKPOINT_REQUEST_FILE"
    if [[ "$AUTO_RESUBMIT" != 1 ]]; then
        echo "Automatic continuation is disabled; submit again to resume."
        exit 0
    fi
    echo "Checkpoint complete; submitting a successor to job ${SLURM_JOB_ID}."
    if ! SUCCESSOR_SUBMISSION=$(
        /usr/bin/sbatch --dependency="afterany:${SLURM_JOB_ID}" \
            /e/home/jusers/bhandari1/jupiter/improvnet_melody_contrastive/run_scripts/melody_launch.sh
    ); then
        echo "Failed to submit the successor job." >&2
        exit 1
    fi
    echo "$SUCCESSOR_SUBMISSION"
    exit 0
}
trap submit_successor_after_checkpoint USR1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

srun --unbuffered --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node=4 \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --module improvnet.model.melody_train &
TRAINING_STEP_PID=$!

wait "$TRAINING_STEP_PID"
TRAINING_STATUS=$?
rm -f "$CHECKPOINT_REQUEST_FILE"
exit "$TRAINING_STATUS"
