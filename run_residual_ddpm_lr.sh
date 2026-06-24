#!/bin/bash
source "$1"
mkdir -p "${OUTPUT_PATH}"

QOS_LINE=""
if [ -n "${QOS:-}" ] ; then
    QOS_LINE="#SBATCH --qos=${QOS}"
fi

MAIL_LINE=""
if [ -n "${MAIL:-}" ] ; then
    MAIL_LINE="#SBATCH --mail-user=${MAIL}"
fi

sbatch << EOT
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
${QOS_LINE}
#SBATCH --time=${TIME}
#SBATCH -N 1
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${N_GPU}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
${MAIL_LINE}
#SBATCH -o ${OUTPUT_PATH}/run.out
#SBATCH -e ${OUTPUT_PATH}/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

source ${SOURCE_PATH}
conda activate ${ENV_PATH}

cd ${MAIN_PATH}

python train_residual_ddpm_lr.py \
    --gnn_pred_file=${GNN_PRED_FILE} \
    --target_file=${TARGET_FILE} \
    --time_index_file=${TIME_INDEX_FILE} \
    --low_input_file=${LOW_INPUT_FILE} \
    --graph_file=${GRAPH_FILE} \
    --static_file=${STATIC_FILE} \
    --train_years=${TRAIN_YEARS:-} \
    --val_years=${VAL_YEARS:-} \
    --output_path=${OUTPUT_PATH} \
    --log_file=${LOG_FILE} \
    --target_type=${TARGET_TYPE} \
    --epochs=${EPOCHS} \
    --batch_size=${BATCH_SIZE} \
    --lr=${LR} \
    --weight_decay=${WEIGHT_DECAY} \
    --unet_base=${UNET_BASE} \
    --lr_hidden=${LR_HIDDEN} \
    --time_emb_dim=${TIME_EMB_DIM} \
    --ddpm_timesteps=${DDPM_TIMESTEPS} \
    --beta_start=${BETA_START} \
    --beta_end=${BETA_END} \
    --grid_h=${GRID_H} \
    --grid_w=${GRID_W} \
    ${SPATIAL_CONFIDENCE_GATE_FLAG:-"--no-use_spatial_confidence_gate"} \
    --gate_smooth_kernel=${GATE_SMOOTH_KERNEL:-9} \
    --gate_threshold=${GATE_THRESHOLD:-1.0} \
    --gate_tau=${GATE_TAU:-0.5} \
    --gate_min=${GATE_MIN:-0.0} \
    --lambda_recon=${LAMBDA_RECON} \
    --lambda_wet=${LAMBDA_WET} \
    --wet_threshold=${WET_THRESHOLD} \
    --aux_residual_model_clip=${AUX_RESIDUAL_MODEL_CLIP} \
    --aux_log_clip=${AUX_LOG_CLIP} \
    --checkpoint_interval=${CHECKPOINT_INTERVAL:-10} \
    --val_seed=${VAL_SEED:-12345} \
    --val_repeats=${VAL_REPEATS:-1} \
    ${NORMALIZE_RESIDUAL_FLAG} \
    ${USE_EMA_FLAG} \
    --ema_decay=${EMA_DECAY} \
    --seed=${SEED}
EOT
