#!/bin/bash
source "$1"

mkdir -p "${LOG_PATH}"

sbatch << EOT
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH --time=${TIME}
#SBATCH -N 1
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${N_GPU}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

# Optional source
[[ -n "${SOURCE_PATH}" ]] && source "${SOURCE_PATH}"

conda activate "${ENV_PATH}"

cd "${MAIN_PATH}"
export PYTHONPATH="\$(pwd):\$PYTHONPATH"
export CARTOPY_DATA_DIR="${CARTOPY_DATA_DIR}"

# Build dynamic argument list
ARGS=()

add_arg() {
    local name="\$1"
    local value="\$2"
    [[ -n "\$value" ]] && ARGS+=( "--\${name}=\${value}" )
}

add_flag() {
    local flag="\$1"
    [[ -n "\$flag" ]] && ARGS+=( "\$flag" )
}

# Boolean flag
add_flag "${USE_ACCELERATE}"

# Simple key=value args
add_arg input_path "${INPUT_PATH}"
add_arg output_path "${OUTPUT_PATH}"
add_arg log_file "${LOG_FILE}"
add_arg graph_file "${GRAPH_FILE}"
add_arg dataset_name "${DATASET_NAME}"
add_arg output_file "${OUTPUT_FILE}"

# Test period
add_arg test_year_start "${TEST_YEAR_START}"
add_arg test_month_start "${TEST_MONTH_START}"
add_arg test_day_start "${TEST_DAY_START}"
add_arg test_year_end "${TEST_YEAR_END}"
add_arg test_month_end "${TEST_MONTH_END}"
add_arg test_day_end "${TEST_DAY_END}"

# Test years (string)
[[ -n "${TEST_YEARS}" ]] && ARGS+=( "--test_years=${TEST_YEARS}" )

# Fixed batch size
ARGS+=( "--batch_size=1" )

# Model + seed
add_arg model_name "${MODEL_NAME}"
add_arg seed "${SEED}"

# Training path + checkpoint
add_arg train_path "${TRAIN_PATH}"
add_arg epoch "${EPOCH}"
add_arg checkpoint "${CHECKPOINT}"

# Target + predictors
add_arg target_type "${TARGET_TYPE}"
add_arg target_file "${TARGET_FILE}"
add_arg low_input_file "${LOW_INPUT_FILE}"
add_arg orog_file "${OROG_FILE}"
add_arg mask_sealand_file "${MASK_SEALAND_FILE}"
add_arg coords_ij_file "${COORDS_IJ_FILE}"
add_arg metadata_file "${METADATA_FILE}"

# Loss + history + threshold
add_arg loss_name "${LOSS_NAME}"
add_arg history_length "${HISTORY_LENGTH}"
add_arg threshold_nll "${THRESHOLD_NLL}"
add_arg threshold "${THRESHOLD}"

# Debug print
echo "ARGS: \${ARGS[@]}"

# Run prediction
accelerate launch \
    --config_file "${ACCELERATE_CONFIG_PATH}" \
    -m predict.predict \
    "\${ARGS[@]}"

# Plot report
python -m utils.plotting.plot_report \
    --input_path="${OUTPUT_PATH}" \
    --plot_path="${OUTPUT_PATH}" \
    --val_file="${OUTPUT_FILE}" \
    --val_file_help="${VAL_FILE_HELP}" \
    --var="${VAR}" \
    --experiment="${EXPERIMENT}" \
    --domain="${DOMAIN}" \
    --val_year="${VAL_YEAR}" \
    --config_file="${CONFIG_FILE_VAL_REPORT}" \
    --predictions_multiplier="${PREDICTIONS_MULTIPLIER}" \
    --target_multiplier="${TARGET_MULTIPLIER}"

EOT
