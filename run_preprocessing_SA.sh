#!/bin/bash
source $1
mkdir -p ${LOG_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ict26_esp_0
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=${TIME}
#SBATCH -N 1
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wtang@ictp.it
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

source ${SOURCE_PATH}

conda activate ${ENV_PATH}

cd ${MAIN_PATH}

if [ ${PERFORM_PHASE_2} = true ] ; then
    PYTHONPATH=. python -m preprocessing.preprocessing_SA \
        --output_path=${OUTPUT_PATH_PHASE_2} \
        --log_file=${LOG_FILE} \
        --input_path_predictors=${INPUT_PATH_PREDICTORS} \
        --predictors_file=${PREDICTORS_FILE} \
        --input_path_target=${INPUT_PATH_TARGET} \
        --target_file=${TARGET_FILE} \
        --target_type=${TARGET_TYPE} \
        --target_multiplier=${TARGET_MULTIPLIER} \
        --input_path_topo=${INPUT_PATH_TOPO} \
        --topo_file=${TOPO_FILE} \
        --lon_min=${LON_MIN} \
        --lon_max=${LON_MAX} \
        --lat_min=${LAT_MIN} \
        --lat_max=${LAT_MAX} \
        --lon_grid_radius_high=${LON_GRID_RADIUS_HIGH} \
        --lat_grid_radius_high=${LAT_GRID_RADIUS_HIGH} \
        --predictors_dataset=${PREDICTORS_DATASET} \
        --target_dataset=${TARGET_DATASET}
fi
EOT