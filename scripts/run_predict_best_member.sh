#!/bin/bash
source "$1"

mkdir -p "${LOG_PATH}"

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
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

[[ -n "${SOURCE_PATH}" ]] && source "${SOURCE_PATH}"

conda activate "${ENV_PATH}"

cd "${MAIN_PATH}"
export PYTHONPATH="\$(pwd):\$PYTHONPATH"
export CARTOPY_DATA_DIR="${CARTOPY_DATA_DIR}"

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

add_flag "${USE_ACCELERATE}"

add_arg input_path "${INPUT_PATH}"
add_arg output_path "${OUTPUT_PATH}"
add_arg log_file "${LOG_FILE}"
add_arg graph_file "${GRAPH_FILE}"
add_arg dataset_name "${DATASET_NAME}"
add_arg output_file "${OUTPUT_FILE}"

add_arg test_year_start "${TEST_YEAR_START}"
add_arg test_month_start "${TEST_MONTH_START}"
add_arg test_day_start "${TEST_DAY_START}"
add_arg test_year_end "${TEST_YEAR_END}"
add_arg test_month_end "${TEST_MONTH_END}"
add_arg test_day_end "${TEST_DAY_END}"
[[ -n "${TEST_YEARS}" ]] && ARGS+=( "--test_years=${TEST_YEARS}" )

ARGS+=( "--batch_size=1" )

add_arg model_name "${MODEL_NAME}"
add_arg seed "${SEED}"
add_arg train_path "${TRAIN_PATH}"
add_arg epoch "${EPOCH}"
add_arg checkpoint "${CHECKPOINT}"
add_arg target_type "${TARGET_TYPE}"
add_arg target_file "${TARGET_FILE}"
add_arg low_input_file "${LOW_INPUT_FILE}"
add_arg orog_file "${OROG_FILE}"
add_arg mask_sealand_file "${MASK_SEALAND_FILE}"
add_arg coords_ij_file "${COORDS_IJ_FILE}"
add_arg metadata_file "${METADATA_FILE}"
add_arg loss_name "${LOSS_NAME}"
add_arg history_length "${HISTORY_LENGTH}"
add_arg n_steps "${N_STEPS}"
add_arg n_samples "${N_SAMPLES}"
add_arg threshold "${THRESHOLD}"
add_arg best_member_transform "${BEST_MEMBER_TRANSFORM}"
add_arg best_member_reference_quantile "${BEST_MEMBER_REFERENCE_QUANTILE}"
add_arg best_member_tail_quantile "${BEST_MEMBER_TAIL_QUANTILE}"
add_arg best_member_tail_target "${BEST_MEMBER_TAIL_TARGET}"
add_arg best_member_tail_weight "${BEST_MEMBER_TAIL_WEIGHT}"
add_arg best_member_tail_excess_weight "${BEST_MEMBER_TAIL_EXCESS_WEIGHT}"

echo "ARGS: \${ARGS[@]}"

accelerate launch \
    --config_file "${ACCELERATE_CONFIG_PATH}" \
    -m predict.predict \
    "\${ARGS[@]}"

if [[ "${VAR}" == "pr" ]]; then
    PRED_ATTR="pr_gnn4cd_best_member"
else
    PRED_ATTR="tasmax_gnn4cd_best_member"
fi

python ./utils/plotting/plot_report.py \
    --input_path="${OUTPUT_PATH}" \
    --plot_path="${OUTPUT_PATH}" \
    --val_file="${OUTPUT_FILE}" \
    --var="${VAR}" \
    --experiment="ESD_pseudo_reality" \
    --val_year="${TEST_YEAR_START}" \
    --domain="${DOMAIN}" \
    --config_file="${CONFIG_FILE_VAL_REPORT}" \
    --pred_attr="\${PRED_ATTR}"

EOT
