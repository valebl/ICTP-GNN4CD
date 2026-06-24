#!/bin/bash
source "$1"
mkdir -p "${REPORT_OUTPUT_PATH}"

# Match the training/prediction environment as closely as possible. The test
# prediction files are PyG objects, so loading them can require the same gcc /
# conda runtime used during inference.
if command -v module >/dev/null 2>&1; then
    module purge
    module load --auto profile/deeplrn
    module load gcc
    module load cuda/11.8
fi

SOURCE_PATH="${SOURCE_PATH:-/leonardo/home/userexternal/${USER}/Conda_init.txt}"
ENV_PATH="${ENV_PATH:-/leonardo/pub/userexternal/sdigioia/sdigioia/env/RLenv}"
if [ -f "${SOURCE_PATH}" ]; then
    source "${SOURCE_PATH}"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate "${ENV_PATH}"
fi

DELTA_ARGS=""
if [ -n "${DELTA_HISTORICAL_FILE:-}" ]; then
    DELTA_ARGS="${DELTA_ARGS} --delta-historical-file=${DELTA_HISTORICAL_FILE}"
fi
if [ -n "${DELTA_MID_CENTURY_FILE:-}" ]; then
    DELTA_ARGS="${DELTA_ARGS} --delta-mid-century-file=${DELTA_MID_CENTURY_FILE}"
fi
if [ -n "${DELTA_END_CENTURY_FILE:-}" ]; then
    DELTA_ARGS="${DELTA_ARGS} --delta-end-century-file=${DELTA_END_CENTURY_FILE}"
fi

DEFAULT_PYTHON="/leonardo/pub/userexternal/sdigioia/sdigioia/env/RLenv/bin/python3.10"
if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN="${PYTHON}"
elif [ -x "${DEFAULT_PYTHON}" ]; then
    PYTHON_BIN="${DEFAULT_PYTHON}"
else
    PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" "${MAIN_PATH}/utils/plotting/plot_test_extreme_report.py" \
    --historical-file="${HISTORICAL_FILE}" \
    --mid-century-file="${MID_CENTURY_FILE}" \
    --end-century-file="${END_CENTURY_FILE}" \
    ${DELTA_ARGS} \
    --output-path="${REPORT_OUTPUT_PATH}" \
    --report-name="${REPORT_NAME}" \
    --daily-date="${DAILY_DATE:-1998-11-13}" \
    --field-name="${FIELD_NAME:-}" \
    --grid-h="${GRID_H:-128}" \
    --grid-w="${GRID_W:-128}" \
    --daily-vmax="${DAILY_VMAX:-225}" \
    --rx1day-vmax="${RX1DAY_VMAX:-225}" \
    --delta-vmax="${DELTA_VMAX:-45}" \
    --delta-mid-vmax="${DELTA_MID_VMAX:-${DELTA_VMAX:-45}}" \
    --delta-end-vmax="${DELTA_END_VMAX:-${DELTA_VMAX:-45}}"
