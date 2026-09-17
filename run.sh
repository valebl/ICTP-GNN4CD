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
# SBATCH --mail-user=${MAIL}
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
python -m accelerate.commands.launch --config_file ${ACCELERATE_CONFIG_PATH} main.py --input_path=${INPUT_PATH} --output_path=${OUTPUT_PATH} --log_file=${LOG_FILE} --target_file=${TARGET_FILE} --graph_file=${GRAPH_FILE} --out_checkpoint_file=${OUT_CHECKPOINT_FILE} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} --step_size=${LR_STEP_SIZE} --lr=${LR} --weight_decay=${WEIGHT_DECAY} --loss_fn=${LOSS_FN} --model_type=${MODEL_TYPE} --model_name=${MODEL_NAME} --dataset_name=${DATASET_NAME} --collate_name=${COLLATE_NAME} --wandb_project_name=${WANDB_PROJECT_NAME} ${USE_ACCELERATE} ${CTD_TRAINING} ${FINE_TUNING} --train_year_start=${TRAIN_YEAR_START} --train_month_start=${TRAIN_MONTH_START} --train_day_start=${TRAIN_DAY_START} --train_year_end=${TRAIN_YEAR_END} --train_month_end=${TRAIN_MONTH_END} --train_day_end=${TRAIN_DAY_END} --first_year=${FIRST_YEAR} --validation_year=${VALIDATION_YEAR} --out_checkpoint_file=${OUT_CHECKPOINT_FILE} --seed=${SEED} --alpha=${ALPHA} --lr_scheduler=${LR_SCHEDULER} --seq_l=${SEQ_L} ${MAKE_VAL_PLOTS} --wandb_api_key={WANDB_API_KEY} --wandb_username={WANDB_USERNAME}
EOT

