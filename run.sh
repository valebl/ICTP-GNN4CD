#!/bin/bash
source $1
mkdir -p ${LOG_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ict26_esp_0
#SBATCH -p boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --time=${TIME}
#SBATCH -N 1
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${N_GPU}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=${MAIL}
#SBATCH -o ${LOG_PATH}run.out
#SBATCH -e ${LOG_PATH}run.err

source ${SOURCE_PATH}

module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

conda activate ${ENV_PATH}

export TMPDIR=/leonardo_work/ICT26_ESP_0/wtang/tmp_slurm_\${SLURM_JOB_ID}
mkdir -p \${TMPDIR}

cd ${MAIN_PATH}

python -m accelerate.commands.launch --config_file ${ACCELERATE_CONFIG_PATH} main.py \
    --input_path=${INPUT_PATH} \
    --output_path=${OUTPUT_PATH} \
    --log_file=${LOG_FILE} \
    --low_input_file=${LOW_INPUT_FILE} \
    --orog_file=${OROG_FILE} \
    --mask_sealand_file=${MASK_SEALAND_FILE} \
    --coords_ij_file=${COORDS_IJ_FILE} \
    --target_file=${TARGET_FILE} \
    --graph_file=${GRAPH_FILE} \
    --metadata_file=${METADATA_FILE} \
    --dataset_name=${DATASET_NAME} \
    --collate_name=${COLLATE_NAME} \
    --epochs=${EPOCHS} \
    --batch_size=${BATCH_SIZE} \
    --step_size=${LR_STEP_SIZE} \
    --gamma=${LR_GAMMA} \
    --lr=${LR} \
    --weight_decay=${WEIGHT_DECAY} \
    --loss_fn=${LOSS_FN} \
    --alpha=${ALPHA} \
    --beta=${BETA} \
    --balance=${BALANCE} \
    --threshold=${THRESHOLD:-0.0} \
    --binmin=${BINMIN:-0.0} \
    --binmax=${BINMAX:-1000} \
    --binwidth=${BINWIDTH:-0.5} \
    --binscale=${BINSCALE:-log} \
    --model_type=${MODEL_TYPE} \
    --model_name=${MODEL_NAME} \
    --target_type=${TARGET_TYPE} \
    --norm_mode=${NORM_MODE:-minmax} \
    --run_type=${RUN_TYPE} \
    --lr_scheduler=${LR_SCHEDULER} \
    --seed=${SEED} \
    --n_gpu=${N_GPU} \
    --wandb_project_name=${WANDB_PROJECT_NAME} \
    --train_year_start=${TRAIN_YEAR_START} \
    --train_month_start=${TRAIN_MONTH_START} \
    --train_day_start=${TRAIN_DAY_START} \
    --train_year_end=${TRAIN_YEAR_END} \
    --train_month_end=${TRAIN_MONTH_END} \
    --train_day_end=${TRAIN_DAY_END} \
    --validation_year=${VALIDATION_YEAR} \
    --first_year=${FIRST_YEAR} \
    --last_year=${LAST_YEAR} \
    --n_val_years=${N_VAL_YEARS} \
    --train_years="${TRAIN_YEARS}" \
    --val_years="${VAL_YEARS}" \
    --checkpoint_ctd=${CHECKPOINT_CTD} \
    ${USE_ACCELERATE} \
    ${CTD_TRAINING} \
    ${FINE_TUNING} \
    ${MAKE_VAL_PLOTS}
EOT