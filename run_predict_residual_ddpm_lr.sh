#!/bin/bash
source "$1"

submit_prediction() {
    mkdir -p "${OUTPUT_PATH}"

    if [ "${APPLY_COARSE_CONSERVATION}" = true ] ; then
        COARSE_FLAG="--apply_coarse_conservation"
    else
        COARSE_FLAG="--no-apply_coarse_conservation"
    fi

    TEST_PREDICTOR_ARGS=""
    if [ -n "${TEST_INPUT_PATH_P:-}" ] && [ -n "${PREDICTORS_FILE:-}" ] ; then
        TEST_PREDICTOR_ARGS="--test_input_path_p=${TEST_INPUT_PATH_P} --predictors_filename=${PREDICTORS_FILE} --input_graph_path=${INPUT_GRAPH_PATH:-}"
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
#SBATCH -o ${OUTPUT_PATH}/${JOB_NAME}.out
#SBATCH -e ${OUTPUT_PATH}/${JOB_NAME}.err

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
    ${TEST_PREDICTOR_ARGS} \
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
    ${SPATIAL_CONFIDENCE_GATE_FLAG:-"--no-use_spatial_confidence_gate"} \
    --gate_smooth_kernel=${GATE_SMOOTH_KERNEL:-9} \
    --gate_threshold=${GATE_THRESHOLD:-1.0} \
    --gate_tau=${GATE_TAU:-0.5} \
    --gate_min=${GATE_MIN:-0.0} \
    --ensemble_method=${ENSEMBLE_METHOD:-raw} \
    --ensemble_smooth_kernel=${ENSEMBLE_SMOOTH_KERNEL:-9} \
    --coarse_block_size=${COARSE_BLOCK_SIZE} \
    --coarse_scale_min=${COARSE_SCALE_MIN} \
    --coarse_scale_max=${COARSE_SCALE_MAX} \
    --noise_scale=${NOISE_SCALE} \
    --reverse_noise_scale=${REVERSE_NOISE_SCALE} \
    --residual_scale=${RESIDUAL_SCALE:-1.0} \
    --snapshot_interval=${SNAPSHOT_INTERVAL} \
    --seed=${SEED} \
    --year_filter=${YEAR_FILTER} \
    ${COARSE_FLAG}

if [ "${MAKE_REPORT:-true}" = true ] && [ -f "${OUTPUT_PATH}${OUTPUT_FILE}" ]; then
    cd ${VALENTINA_PATH}
    python ./utils/plotting/plot_report.py \
        --input_path=${OUTPUT_PATH} \
        --plot_path=${OUTPUT_PATH} \
        --val_file="${OUTPUT_FILE}" \
        --var=${VAR} \
        --experiment="${EXPERIMENT}" \
        --val_year=${TEST_YEAR_START} \
        --domain=SA
elif [ "${MAKE_REPORT:-true}" = true ]; then
    echo "ERROR: ${OUTPUT_PATH}${OUTPUT_FILE} not found, skipping plot_report.py"
else
    echo "MAKE_REPORT=false, skipping standard plot_report.py"
fi

if [ "${MAKE_REPORT:-true}" = true ] && [ ${SNAPSHOT_INTERVAL} -gt 0 ]; then
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
}

if [ "${TEST_MATRIX:-false}" = true ] ; then
    for model_spec in "${TEST_MODELS[@]}"; do
        IFS='|' read -r MODEL_TAG MODEL_NETCDF_TAG <<< "${model_spec}"
        for period_spec in "${TEST_PERIODS[@]}"; do
            IFS='|' read -r PERIOD PERIOD_YEARS PERIOD_START <<< "${period_spec}"
            for INPUT_MODE in "${TEST_INPUT_MODES[@]}"; do
                GNN_PRED_FILE="${ATTENTION_TEST_BASE}/${PERIOD}/${INPUT_MODE}/${MODEL_TAG}_pr_attention.pkl"
                TEST_INPUT_PATH_P="${TEST_PREDICTOR_BASE}/${PERIOD}/predictors/${INPUT_MODE}/"
                PREDICTORS_FILE="${MODEL_NETCDF_TAG}_${PERIOD_YEARS}.nc"
                OUTPUT_PATH="${DDPM_TEST_OUTPUT_BASE}/${PERIOD}/${INPUT_MODE}/"
                OUTPUT_FILE="${MODEL_TAG}_pr_attention_ddpm_recon_wet.pkl"
                LOG_FILE="${MODEL_TAG}_log.txt"
                TEST_YEAR_START="${PERIOD_START}"
                JOB_NAME="${JOB_NAME_PREFIX}-${MODEL_TAG}-${PERIOD}-${INPUT_MODE}"

                echo "Submitting ${MODEL_TAG} ${PERIOD} ${INPUT_MODE}"
                submit_prediction
            done
        done
    done
else
    submit_prediction
fi
