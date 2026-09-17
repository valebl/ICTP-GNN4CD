#!/bin/bash
source $1
mkdir -p ${LOG_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ict25_esp_0
#SBATCH -p boost_usr_prod
# SBATCH --qos=qos_prio
#SBATCH --qos=boost_qos_dbg
# SBATCH --time=00:30:00        # format: HH:MM:SS
#SBATCH --time ${TIME}       # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1   # 8 tasks out of 128
#SBATCH --job-name=${JOB_NAME}
# SBATCH --mail-type=FAIL,END
# SBATCH --mail-user=${MAIL}
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

#----------#
# PHASE 1  #
#----------#
module purge
module load --auto profile/meteo
module load cdo/2.1.0--gcc--11.3.0

source ${SOURCE_PATH}

cd ${INPUT_PATH_PHASE_1}

if [ ${PERFORM_PHASE_1} = true ] ; then
	source ${PHASE_1_PATH} ${LON_MIN} ${LON_MAX} ${LAT_MIN} ${LAT_MAX} ${INTERVAL} ${INPUT_PATH_PHASE_1} ${OUTPUT_PATH_PHASE_1} ${PREFIX_PHASE_1}
fi

#---------#
# PHASE 2 #
#---------#

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

conda activate ${ENV_PATH}

cd ${MAIN_PATH}

if [ ${PERFORM_PHASE_2} = true ] ; then
        PYTHONPATH=. python -m preprocessing.${PHASE_2_PYTHON_FILE} --input_path_phase_2=${INPUT_PATH_PHASE_2} --input_path_gripho=${INPUT_PATH_GRIPHO} --input_path_topo=${INPUT_PATH_TOPO} --gripho_file=${GRIPHO_FILE} --topo_file=${TOPO_FILE} --output_path=${OUTPUT_PATH_PHASE_2} --log_file=${LOG_FILE} --lon_min=${LON_MIN} --lon_max=${LON_MAX} --lat_min=${LAT_MIN} --lat_max=${LAT_MAX} --suffix_phase_2=${SUFFIX_PHASE_2} --predictors_type=${PREDICTORS_TYPE} --lon_grid_radius_high=${LON_GRID_RADIUS_HIGH} --lat_grid_radius_high=${LAT_GRID_RADIUS_HIGH} --mask_path=${MASK_PATH} --mask_file=${MASK_FILE} --land_use_path=${LAND_USE_PATH} --land_use_file=${LAND_USE_FILE} --target_multiplier=${TARGET_MULTIPLIER}
fi
EOT


