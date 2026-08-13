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
add_arg target_variables "${TARGET_VARIABLES}"
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
add_arg n_members "${N_MEMBERS}"

# Debug print
echo "ARGS: \${ARGS[@]}"

# Run prediction
accelerate launch \
    --config_file "${ACCELERATE_CONFIG_PATH}" \
    -m predict.predict \
    "\${ARGS[@]}"

# Plot report(s) -- one per target variable. Read the variable list back
# from OUTPUT_PATH/target_variables.json, which predict.py writes out as the
# definitive resolved order/set it actually predicted (may differ from
# TARGET_VARIABLES above if predict.py fell back to a training-recorded
# order) -- avoids the report ever drifting from what was actually predicted.
TARGET_VARS_RESOLVED=\$(python3 -c "import json; print(','.join(json.load(open('${OUTPUT_PATH}target_variables.json'))['target_variables']))")
IFS=',' read -ra VAR_ARR <<< "\$TARGET_VARS_RESOLVED"

for VAR_I in "\${VAR_ARR[@]}"; do
    echo "Generating report for variable: \${VAR_I}"
    python -m utils.plotting.plot_report \
        --input_path="${OUTPUT_PATH}" \
        --plot_path="${OUTPUT_PATH}" \
        --val_file="${OUTPUT_FILE}" \
        --val_file_help="${VAL_FILE_HELP}" \
        --var="\${VAR_I}" \
        --experiment="${EXPERIMENT}" \
        --domain="${DOMAIN}" \
        --val_year="${VAL_YEAR}" \
        --config_file="${CONFIG_FILE_VAL_REPORT}" \
        --predictions_multiplier="${PREDICTIONS_MULTIPLIER}" \
        --target_multiplier="${TARGET_MULTIPLIER}"
done

# Aggregated spatial-means report -- one PDF, one page, all target
# variables' Ground-Truth/Prediction time-mean maps together. Multivariate
# runs only (a single-variable run already has this via plot_report.py's
# own Average page). Reuses the same resolved variable list as the
# per-variable loop above. VARIABLE_ORDER (optional, comma-separated, same
# set as TARGET_VARIABLES) overrides the row-fill display order on the page.
if [[ "\${#VAR_ARR[@]}" -gt 1 ]]; then
    echo "Generating aggregated spatial-means report"
    AGG_ARGS=(
        --input_path="${OUTPUT_PATH}"
        --val_file="${OUTPUT_FILE}"
        --plot_path="${OUTPUT_PATH}"
        --target_variables="\${TARGET_VARS_RESOLVED}"
        --config_file="${CONFIG_FILE_VAL_REPORT}"
        --domain="${DOMAIN}"
        --experiment="${EXPERIMENT}"
        --val_year="${VAL_YEAR}"
    )
    [[ -n "${VARIABLE_ORDER}" ]] && AGG_ARGS+=( --variable_order="${VARIABLE_ORDER}" )
    python -m utils.plotting.plot_aggregated_means "\${AGG_ARGS[@]}"
fi

EOT