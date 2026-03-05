#!/bin/bash
source $1
mkdir -p ${LOG_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ict26_esp_0
#SBATCH -p boost_usr_prod
# SBATCH --qos=qos_prio
#SBATCH --qos=boost_qos_dbg
# SBATCH --time=00:30:00        # format: HH:MM:SS
#SBATCH --time ${TIME}       # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1   # 8 tasks out of 128
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wtang@ictp.it
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

#----------#
# PHASE 1  #
#----------#
module purge
module load --auto profile/meteo
module load --auto profile/deeplrn
module load gcc
module load cdo/2.1.0--gcc--11.3.0
module load cuda/11.8

source ${SOURCE_PATH}

conda activate ${ENV_PATH}

cd ${MAIN_PATH}

if [ ${PERFORM_PHASE_2} = true ] ; then
        PYTHONPATH=. python -m preprocessing.${PHASE_2_PYTHON_FILE} --input_path_target=${INPUT_PATH_TARGET} --input_path_topo=${INPUT_PATH_TOPO} --target_file=${TARGET_FILE} --topo_file=${TOPO_FILE} --output_path=${OUTPUT_PATH_PHASE_2} --log_file=${LOG_FILE} --lon_min=${LON_MIN} --lon_max=${LON_MAX} --lat_min=${LAT_MIN} --lat_max=${LAT_MAX} --suffix_phase_2=${SUFFIX_PHASE_2} --predictors_type=${PREDICTORS_TYPE} --lon_grid_radius_high=${LON_GRID_RADIUS_HIGH} --lat_grid_radius_high=${LAT_GRID_RADIUS_HIGH} --domain=${DOMAIN} --experiment=${EXPERIMENT} --target_multiplier=${TARGET_MULTIPLIER}
fi
EOT


