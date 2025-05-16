#!/bin/bash
#######################################################################################################################
#
# This script runs the evaluation for a RWKV soft-label model.
# Ensure that the model parameters (N_LAYER, N_EMBD, etc.) match the checkpoint being evaluated.
#
#######################################################################################################################
#
MODEL_TYPE="x070" # x060 => rwkv-6.0 (Ensure this matches the model architecture)
#
N_LAYER="12"
N_EMBD="768"
#
CTX_LEN="4096"
#
#######################################################################################################################
# Evaluation Specific Parameters
#######################################################################################################################
#
CHECKPOINT_PATH="/public/home/ssjxzkz/Projects/rwkv-cc/out/L12-D768-x070/rwkv-80.pth"
# !!! PLEASE SPECIFY THE PATH TO YOUR EVALUATION DATA FILE !!!
DATA_FILE_PATH="/public/home/ssjxzkz/Datasets/prot/ncbi_nr/mmap/softlabel"
DATA_TYPE="binidx" # Or your specific data type if different for eval
#
M_BSZ="16" # Micro batch size for evaluation (can be adjusted based on VRAM)
PRECISION="bf16" # bf16, fp16, fp32, tf32
DEVICE="cuda" # cuda or cpu
VOCAB_SIZE="65" # Vocab size MyDataset might expect, model head is 64
HEAD_SIZE="64"
#
N_NODE_EVAL="1"    # Number of nodes for evaluation
DEVICES_EVAL="1" # Number of devices per node for evaluation
#
# !!! SET MAGIC_PRIME: This value MUST match the one used during training for the specific dataset and CTX_LEN !!!
# Example value from demo-training-run.sh, replace with your actual magic_prime for the data being evaluated
MAGIC_PRIME_EVAL="9617999" 
TRAIN_STAGE_EVAL="2" # Set to 0 for basic eval, or another stage if MyDataset behavior depends on it for eval.

#######################################################################################################################
# Environment Setup (modify as needed)
#######################################################################################################################
#
# Assuming the script is run from the project root or paths are adjusted accordingly.
# Activate your virtual environment if you have one
if [ -f "/public/home/ssjxzkz/Projects/rwkv-cc/.venv/bin/activate" ]; then
    source "/public/home/ssjxzkz/Projects/rwkv-cc/.venv/bin/activate"
else
    echo "Virtual environment not found at /public/home/ssjxzkz/Projects/rwkv-cc/.venv/bin/activate"
fi

# Change to the project directory
PROJECT_ROOT="/public/home/ssjxzkz/Projects/rwkv-cc"
cd "$PROJECT_ROOT"

# Set PYTHONPATH to include the project's src directory
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Set WANDB_MODE if needed (e.g., offline if not logging to wandb for eval)
export WANDB_MODE=offline

#######################################################################################################################
# Run Evaluation Script
#######################################################################################################################

echo "Starting evaluation script..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Evaluation Data: $DATA_FILE_PATH"

python "$PROJECT_ROOT/run/eval.py" \
    --n_layer "$N_LAYER" \
    --n_embd "$N_EMBD" \
    --ctx_len "$CTX_LEN" \
    --vocab_size "$VOCAB_SIZE" \
    --head_size "$HEAD_SIZE" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_file "$DATA_FILE_PATH" \
    --data_type "$DATA_TYPE" \
    --precision "$PRECISION" \
    --micro_bsz "$M_BSZ" \
    --device "$DEVICE" \
    --my_testing "$MODEL_TYPE" \
    --num_nodes "$N_NODE_EVAL" \
    --devices "$DEVICES_EVAL" \
    --magic_prime "$MAGIC_PRIME_EVAL" \
    --train_stage "$TRAIN_STAGE_EVAL"

echo "Evaluation script finished."
