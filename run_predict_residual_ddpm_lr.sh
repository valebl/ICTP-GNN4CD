#!/bin/bash
source "$1"
mkdir -p "${OUTPUT_PATH}"

if [ "${APPLY_COARSE_CONSERVATION}" = true ] ; then
    COARSE_FLAG="--apply_coarse_conservation"
else
    COARSE_FLAG="--no-apply_coarse_conservation"
fi

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

python predict_residual_ddpm_lr.py \
    --gnn_pred_file=${GNN_PRED_FILE} \
    --graph_file=${GRAPH_FILE} \
    --static_file=${STATIC_FILE} \
    --low_input_file=${LOW_INPUT_FILE} \
    --time_index_file=${TIME_INDEX_FILE} \
    --lr_norm_file=${LR_NORM_FILE} \
    --residual_norm_file=${RESIDUAL_NORM_FILE} \
    --checkpoint=${CHECKPOINT} \
    --output_path=${OUTPUT_PATH} \
    --output_file=${OUTPUT_FILE} \
    --log_file=${LOG_FILE} \
    --target_type=${TARGET_TYPE} \
    --ddpm_timesteps=${DDPM_TIMESTEPS} \
    --beta_start=${BETA_START} \
    --beta_end=${BETA_END} \
    --unet_base=${UNET_BASE} \
    --lr_hidden=${LR_HIDDEN} \
    --time_emb_dim=${TIME_EMB_DIM} \
    --n_samples=${N_SAMPLES} \
    --batch_size=${BATCH_SIZE} \
    --grid_h=${GRID_H} \
    --grid_w=${GRID_W} \
    --coarse_block_size=${COARSE_BLOCK_SIZE} \
    --coarse_scale_min=${COARSE_SCALE_MIN} \
    --coarse_scale_max=${COARSE_SCALE_MAX} \
    --noise_scale=${NOISE_SCALE} \
    --reverse_noise_scale=${REVERSE_NOISE_SCALE} \
    --snapshot_interval=${SNAPSHOT_INTERVAL} \
    --seed=${SEED} \
    --year_filter=${YEAR_FILTER} \
    ${COARSE_FLAG}

if [ -f "${OUTPUT_PATH}${OUTPUT_FILE}" ]; then
    cd ${VALENTINA_PATH}
    python ./utils/plotting/plot_report.py \
        --input_path=${OUTPUT_PATH} \
        --plot_path=${OUTPUT_PATH} \
        --val_file="${OUTPUT_FILE}" \
        --var=${VAR} \
        --experiment="${EXPERIMENT}" \
        --val_year=${TEST_YEAR_START} \
        --domain=SA
else
    echo "ERROR: ${OUTPUT_PATH}${OUTPUT_FILE} not found, skipping plot_report.py"
fi

if [ ${SNAPSHOT_INTERVAL} -gt 0 ]; then
    for STEP in \$(seq ${SNAPSHOT_INTERVAL} ${SNAPSHOT_INTERVAL} ${DDPM_TIMESTEPS}); do
        STEP_PAD=\$(printf "%04d" \${STEP})
        SNAP_FILE="${OUTPUT_FILE%.*}_step\${STEP_PAD}.${OUTPUT_FILE##*.}"
        SNAP_PLOT_DIR="${OUTPUT_PATH}/step_\${STEP_PAD}/"
        if [ -f "${OUTPUT_PATH}\${SNAP_FILE}" ]; then
            mkdir -p "\${SNAP_PLOT_DIR}"
            cd ${VALENTINA_PATH}
            python ./utils/plotting/plot_report.py \
                --input_path=${OUTPUT_PATH} \
                --plot_path="\${SNAP_PLOT_DIR}" \
                --val_file="\${SNAP_FILE}" \
                --var=${VAR} \
                --experiment="${EXPERIMENT}" \
                --val_year=${TEST_YEAR_START} \
                --domain=SA
        else
            echo "WARNING: ${OUTPUT_PATH}\${SNAP_FILE} not found, skipping snapshot report"
        fi
    done
fi
EOT
