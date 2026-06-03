#!/bin/bash
source "$1"

mkdir -p "${LOG_PATH}"

sbatch << EOT
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --time=${TIME}
#SBATCH -N 1
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${N_GPU}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
#SBATCH -o ${LOG_PATH}run.out
#SBATCH -e ${LOG_PATH}run.err

# Optional source
[[ -n "${SOURCE_PATH}" ]] && source "${SOURCE_PATH}"

module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

conda activate "${ENV_PATH}"

cd "${MAIN_PATH}"
export PYTHONPATH="\$(pwd):\$PYTHONPATH"

# Build argument list
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

# Simple key=value args
add_arg seed "${SEED}"
add_arg input_path "${INPUT_PATH}"
add_arg output_path "${OUTPUT_PATH}"
add_arg log_file "${LOG_FILE}"
add_arg target_file "${TARGET_FILE}"
add_arg graph_file "${GRAPH_FILE}"
add_arg epochs "${EPOCHS}"
add_arg batch_size "${BATCH_SIZE}"
add_arg loss_name "${LOSS_NAME}"
add_arg model_name "${MODEL_NAME}"
add_arg dataset_name "${DATASET_NAME}"
add_arg collate_name "${COLLATE_NAME}"
add_arg wandb_project_name "${WANDB_PROJECT_NAME}"


# Date ranges
add_arg train_year_start "${TRAIN_YEAR_START}"
add_arg train_month_start "${TRAIN_MONTH_START}"
add_arg train_day_start "${TRAIN_DAY_START}"
add_arg train_year_end "${TRAIN_YEAR_END}"
add_arg train_month_end "${TRAIN_MONTH_END}"
add_arg train_day_end "${TRAIN_DAY_END}"

# Other numeric args
add_arg validation_year "${VALIDATION_YEAR}"
add_arg first_year "${FIRST_YEAR}"
add_arg last_year "${LAST_YEAR}"
add_arg n_val_years "${N_VAL_YEARS}"
add_arg checkpoint_ctd "${CHECKPOINT_CTD}"
add_arg alpha "${ALPHA}"
add_arg beta "${BETA}"
add_arg binmin "${BINMIN}"
add_arg binmax "${BINMAX}"
add_arg binwidth "${BINWIDTH}"
add_arg binscale "${BINSCALE}"
add_arg threshold "${THRESHOLD}"
add_arg target_type "${TARGET_TYPE}"
add_arg lr "${LR}"
add_arg lr_scheduler "${LR_SCHEDULER}"
add_arg lr_step_size "${LR_STEP_SIZE}"
add_arg lr_gamma "${LR_GAMMA}"
add_arg lr_mode "${LR_MODE}"
add_arg lr_factor "${LR_FACTOR}"
add_arg lr_patience "${LR_PATIENCE}"
add_arg lr_eta_min "${LR_ETA_MIN}"
add_arg lr_warmup_epochs "${LR_WARMUP_EPOCHS}"
add_arg weight_decay "${WEIGHT_DECAY}"
add_arg x_dim "${X_DIM}"
add_arg y_dim "${Y_DIM}"

# File paths
add_arg low_input_file "${LOW_INPUT_FILE}"
add_arg orog_file "${OROG_FILE}"
add_arg mask_sealand_file "${MASK_SEALAND_FILE}"
add_arg coords_ij_file "${COORDS_IJ_FILE}"
add_arg metadata_file "${METADATA_FILE}"

# Arrays
if [[ \${#TRAIN_YEARS[@]} -gt 0 ]]; then
    ARGS+=( "--train_years" "\${TRAIN_YEARS[@]}" )
fi

if [[ \${#VAL_YEARS[@]} -gt 0 ]]; then
    ARGS+=( "--val_years" "\${VAL_YEARS[@]}" )
fi

# More args
add_arg history_length "${HISTORY_LENGTH}"
add_arg predictand_transform_mode "${PREDICTAND_TRANSFORM_MODE}"
add_arg predictor_low_transform_mode "${PREDICTOR_LOW_TRANSFORM_MODE}"
add_arg predictor_high_transform_mode "${PREDICTOR_HIGH_TRANSFORM_MODE}"
add_arg val_plot_frequency "${VAL_PLOT_FREQUENCY}"
add_arg val_plot_config "${VAL_PLOT_CONFIG}"
add_arg WANDB_API_KEY "${WANDB_API_KEY}"
add_arg WANDB_USERNAME "${WANDB_USERNAME}"

# Boolean flags
add_flag "${USE_ACCELERATE}"
add_flag "${CTD_TRAINING}"
add_flag "${MAKE_VAL_PLOTS}"

# Debug print
echo "ARGS: \${ARGS[@]}"

# Launch training
accelerate launch \
    --config_file "${ACCELERATE_CONFIG_PATH}" \
    -m train.train \
    "\${ARGS[@]}"

EOT
