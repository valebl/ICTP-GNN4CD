#!/bin/bash
source "$1"
mkdir -p "${OUTPUT_PATH}"

sbatch << EOT
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
$( [[ -n "${QOS}" ]] && echo "#SBATCH --qos=${QOS}" )
#SBATCH --time=${TIME}
#SBATCH -N 1
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${N_GPU}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
#SBATCH -o ${OUTPUT_PATH}/run.out
#SBATCH -e ${OUTPUT_PATH}/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

[[ -n "${SOURCE_PATH}" ]] && source "${SOURCE_PATH}"
conda activate "${ENV_PATH}"

cd "${MAIN_PATH}"
export PYTHONPATH="\$(pwd):\$PYTHONPATH"

python train_residual_fm_lr.py \
    --gnn_pred_file="${GNN_PRED_FILE}" \
    --target_file="${TARGET_FILE}" \
    --time_index_file="${TIME_INDEX_FILE}" \
    --low_input_file="${LOW_INPUT_FILE}" \
    --graph_file="${GRAPH_FILE}" \
    --static_file="${STATIC_FILE}" \
    --output_path="${OUTPUT_PATH}" \
    --log_file="${LOG_FILE}" \
    --target_type="${TARGET_TYPE}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --lr="${LR}" \
    --weight_decay="${WEIGHT_DECAY}" \
    --unet_base="${UNET_BASE}" \
    --lr_hidden="${LR_HIDDEN}" \
    --time_emb_dim="${TIME_EMB_DIM}" \
    --grid_h="${GRID_H}" \
    --grid_w="${GRID_W}" \
    --highpass_kernel="${HIGHPASS_KERNEL}" \
    --coarse_block_size="${COARSE_BLOCK_SIZE}" \
    --lambda_psd="${LAMBDA_PSD}" \
    --lambda_quantile="${LAMBDA_QUANTILE}" \
    --lambda_coarse="${LAMBDA_COARSE}" \
    --lambda_wet="${LAMBDA_WET}" \
    --wet_threshold="${WET_THRESHOLD}" \
    --seed="${SEED}"
EOT
