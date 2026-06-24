#!/bin/bash
source "$1"

mkdir -p "${OUTPUT_PATH}"
LOG_PATH="${OUTPUT_PATH}"

SBATCH_QOS_DIRECTIVE=""
[[ -n "${QOS}" ]] && SBATCH_QOS_DIRECTIVE="#SBATCH --qos=${QOS}"
SBATCH_MAIL_DIRECTIVE=""
[[ -n "${MAIL}" ]] && SBATCH_MAIL_DIRECTIVE="#SBATCH --mail-user=${MAIL}"

sbatch << EOT
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
${SBATCH_QOS_DIRECTIVE}
#SBATCH --time ${TIME}
#SBATCH -N 1
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
${SBATCH_MAIL_DIRECTIVE}
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

# Optional source
[[ -n "${SOURCE_PATH}" ]] && source "${SOURCE_PATH}"

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

conda activate "${ENV_PATH}"

cd "${MAIN_PATH}"
export PYTHONPATH="\$(pwd):\$PYTHONPATH"

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

# Input paths
add_arg input_path_predictors "${INPUT_PATH_PREDICTORS}"
add_arg input_path_target "${INPUT_PATH_TARGET}"
add_arg input_path_topo "${INPUT_PATH_TOPO}"
add_arg input_path_mask_sealand "${INPUT_PATH_MASK_SEALAND}"

# File names
add_arg target_file "${TARGET_FILE}"
add_arg predictors_file "${PREDICTORS_FILE}"
add_arg mask_sealand_file "${MASK_SEALAND_FILE}"
add_arg topo_file "${TOPO_FILE}"

# Output + logging
add_arg output_path "${OUTPUT_PATH}"
add_arg log_file "${LOG_FILE}"

# Grid radius
add_arg lon_grid_radius_high "${LON_GRID_RADIUS_HIGH}"
add_arg lat_grid_radius_high "${LAT_GRID_RADIUS_HIGH}"

# Mask + land use
add_arg mask_path "${MASK_PATH}"
add_arg mask_file "${MASK_FILE}"
add_arg land_use_path "${LAND_USE_PATH}"
add_arg land_use_file "${LAND_USE_FILE}"

# Target + dataset
add_arg target_type "${TARGET_TYPE}"
add_arg target_multiplier "${TARGET_MULTIPLIER}"
add_arg dataset_name "${DATASET_NAME}"

# Params + levels (string or array)
[[ -n "${PARAMS}" ]] && ARGS+=( "--params=${PARAMS}" )
[[ -n "${LEVELS}" ]] && ARGS+=( "--levels=${LEVELS}" )

# Debug print
echo "ARGS: \${ARGS[@]}"

# Run preprocessing
python -m preprocess.preprocess "\${ARGS[@]}"

EOT
