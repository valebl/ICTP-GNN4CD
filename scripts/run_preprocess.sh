#!/bin/bash
source "$1"

mkdir -p "${OUTPUT_PATH}"
LOG_PATH="${OUTPUT_PATH}"

sbatch << EOT
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH --time ${TIME}
#SBATCH -N 1
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=${JOB_NAME}
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
add_arg output_file "${OUTPUT_FILE}"

# Output + logging
add_arg output_path "${OUTPUT_PATH}"
add_arg log_file "${LOG_FILE}"

# Grid radius
add_arg lon_grid_radius_high "${LON_GRID_RADIUS_HIGH}"
add_arg lat_grid_radius_high "${LAT_GRID_RADIUS_HIGH}"
add_arg lon_grid_radius_low "${LON_GRID_RADIUS_LOW}"
add_arg lat_grid_radius_low "${LAT_GRID_RADIUS_LOW}"
add_arg k_low2high "${K_LOW2HIGH}"
add_arg edge_norm_constants "${EDGE_NORM_CONSTANTS}"

# Mask + land use
add_arg mask_path "${MASK_PATH}"
add_arg mask_file "${MASK_FILE}"
add_arg land_use_path "${LAND_USE_PATH}"
add_arg land_use_file "${LAND_USE_FILE}"

# Dataset
add_arg dataset_name "${DATASET_NAME}"

# Target variables and multipliers
[[ -n "${TARGET_VARIABLES}" ]] && ARGS+=( "--target_variables=${TARGET_VARIABLES}" )
[[ -n "${TARGET_MULTIPLIERS}" ]] && ARGS+=( "--target_multipliers=${TARGET_MULTIPLIERS}" )

# Params + levels (string or array)
[[ -n "${PARAMS}" ]] && ARGS+=( "--params=${PARAMS}" )
[[ -n "${LEVELS}" ]] && ARGS+=( "--levels=${LEVELS}" )

# Debug print
echo "ARGS: \${ARGS[@]}"

# Run preprocessing
python -m preprocess.preprocess "\${ARGS[@]}"

EOT
