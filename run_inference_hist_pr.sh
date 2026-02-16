#!/bin/bash

# ==================== 读取config文件 ====================
CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash run_inference_pr_batch.sh <config_file>"
    exit 1
fi
source ${CONFIG_FILE}
# ==================== 准备参数 ====================
mkdir -p ${LOG_PATH}

PERIODS_ARGS=""
for p in ${PERIODS}; do PERIODS_ARGS="${PERIODS_ARGS} ${p}"; done

GCMS_ARGS=""
for g in ${GCMS}; do GCMS_ARGS="${GCMS_ARGS} ${g}"; done

MODES_ARGS=""
for m in ${MODES}; do MODES_ARGS="${MODES_ARGS} ${m}"; done

if [ ${USE_ACCELERATE:-true} = true ]; then
    USE_ACCELERATE_FLAG="--use_accelerate"
else
    USE_ACCELERATE_FLAG="--no-use_accelerate"
fi

# ==================== SLURM 提交 ====================
sbatch <<EOF
#!/bin/bash
#SBATCH -A ict25_esp_0
#SBATCH -p boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --time=${TIME}  
#SBATCH -N 1
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${N_GPU}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wtang@ictp.it
#SBATCH -o ${LOG_PATH}run.out
#SBATCH -e ${LOG_PATH}run.err


source ${SOURCE_PATH}
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8
conda activate ${ENV_PATH}

cd ${MAIN_PATH}

python inference_pr_Rall.py \
    --base_graph_path  ${BASE_GRAPH_PATH} \
    --base_output_path ${BASE_OUTPUT_PATH} \
    --train_path_reg   ${TRAIN_PATH} \
    --checkpoint_reg   ${CHECKPOINT_REG} \
    --log_path         ${LOG_PATH} \
    --log_file         ${LOG_FILE} \
    --model_name       ${MODEL_NAME} \
    --dataset_name     ${DATASET_NAME} \
    --seq_l            ${SEQ_L} \
    --batch_size       ${BATCH_SIZE} \
    --seed             ${SEED} \
    --pr_threshold     ${PR_THRESHOLD} \
    --periods          ${PERIODS_ARGS} \
    --gcms             ${GCMS_ARGS} \
    --modes            ${MODES_ARGS} \
    --no-use_accelerate

EOF

echo "Job submitted!"
echo "Logs: ${LOG_PATH}"