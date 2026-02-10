#!/bin/bash
source $1
mkdir -p ${LOG_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ict25_esp_0
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=${TIME}       	# format: HH:MM:SS
#SBATCH -N 1                  	# 1 node
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1 	# out of 128
#SBATCH --gres=gpu:${N_GPU}     # 1 gpus per node out of 4
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wtang@ictp.it
#SBATCH -o ${LOG_PATH}run.out
#SBATCH -e ${LOG_PATH}run.err

source ${SOURCE_PATH}

module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

#source ~/anaconda/etc/profile.d/conda.sh
conda activate ${ENV_PATH}

cd ${MAIN_PATH}

## training
python -m accelerate.commands.launch --config_file ${ACCELERATE_CONFIG_PATH} predictions_Rall_future.py --input_path=${INPUT_PATH} --output_path=${OUTPUT_PATH} --log_path=${LOG_PATH} --output_file=${OUTPUT_FILE} --output_file_season=${OUTPUT_FILE_SEASONS} --log_file=${LOG_FILE} --graph_file=${GRAPH_FILE} --dataset_name=${DATASET_NAME} --validation_year=${VALIDATION_YEAR} --test_month_start=${TEST_MONTH_START} --test_day_start=${TEST_DAY_START} --test_month_end=${TEST_MONTH_END} --test_day_end=${TEST_DAY_END} --batch_size=1 --mode=${MODE} --target_file=${TARGET_FILE} --model_name=${MODEL_NAME} --model_type=${MODEL_TYPE} --seed=${SEED} --train_path_reg=${TRAIN_PATH} --checkpoint_reg=${CHECKPOINT_REG} --stats_mode=${STATS_MODE} --target_type=${TARGET_TYPE} --seq_l=${SEQ_L} ${USE_ACCELERATE} ${MAKE_PLOTS}
EOT

