#!/bin/bash
source "$1"
mkdir -p "${OUTPUT_PATH}"

if [ "${APPLY_COARSE_CONSERVATION}" = true ] ; then
    COARSE_FLAG="--apply_coarse_conservation"
else
    COARSE_FLAG="--no-apply_coarse_conservation"
fi

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

python predict_residual_fm_lr.py \
    --gnn_pred_file="${GNN_PRED_FILE}" \
    --graph_file="${GRAPH_FILE}" \
    --static_file="${STATIC_FILE}" \
    --low_input_file="${LOW_INPUT_FILE}" \
    --time_index_file="${TIME_INDEX_FILE}" \
    --lr_norm_file="${LR_NORM_FILE}" \
    --checkpoint="${CHECKPOINT}" \
    --output_path="${OUTPUT_PATH}" \
    --output_file="${OUTPUT_FILE}" \
    --log_file="${LOG_FILE}" \
    --target_type="${TARGET_TYPE}" \
    --n_inference_steps="${N_INFERENCE_STEPS}" \
    --unet_base="${UNET_BASE}" \
    --lr_hidden="${LR_HIDDEN}" \
    --time_emb_dim="${TIME_EMB_DIM}" \
    --n_samples="${N_SAMPLES}" \
    --batch_size="${BATCH_SIZE}" \
    --grid_h="${GRID_H}" \
    --grid_w="${GRID_W}" \
    --coarse_block_size="${COARSE_BLOCK_SIZE}" \
    --noise_scale="${NOISE_SCALE}" \
    --seed="${SEED}" \
    --year_filter=${YEAR_FILTER:-0} \
    ${COARSE_FLAG}

if [ -f "${OUTPUT_PATH}${OUTPUT_FILE}" ]; then
    cd "${VALENTINA_PATH}"
    python ./utils/plotting/plot_report.py \
        --input_path="${OUTPUT_PATH}" \
        --plot_path="${OUTPUT_PATH}" \
        --val_file="${OUTPUT_FILE}" \
        --var="${VAR}" \
        --experiment="${EXPERIMENT}" \
        --val_year="${TEST_YEAR_START}" \
        --domain=SA
else
    echo "ERROR: ${OUTPUT_PATH}${OUTPUT_FILE} not found, skipping plot_report.py"
fi
EOT
