#!/bin/bash
source "$1"

mkdir -p "${OUTPUT_TEST_PATH}"
LOG_PATH="${OUTPUT_TEST_PATH}/logs"
mkdir -p "${LOG_PATH}"

SBATCH_QOS_DIRECTIVE=""
[[ -n "${QOS}" ]] && SBATCH_QOS_DIRECTIVE="#SBATCH --qos=${QOS}"
SBATCH_MAIL_DIRECTIVE=""
[[ -n "${MAIL}" ]] && SBATCH_MAIL_DIRECTIVE="#SBATCH --mail-user=${MAIL}"

sbatch << EOT
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
${SBATCH_QOS_DIRECTIVE}
#SBATCH --time ${TIME}
#SBATCH -N 1
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
${SBATCH_MAIL_DIRECTIVE}
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

[[ -n "${SOURCE_PATH}" ]] && source "${SOURCE_PATH}"

module purge
module load --auto profile/deeplrn
module load gcc

conda activate "${ENV_PATH}"

cd "${MAIN_PATH}"
export PYTHONPATH="\$(pwd):\$PYTHONPATH"

python -m preprocess.preprocess_test_predictors \\
    --raw-test-path="${RAW_TEST_PATH}" \\
    --output-test-path="${OUTPUT_TEST_PATH}" \\
    --periods="${PERIODS}" \\
    --realizations="${REALIZATIONS}" \\
    --params="${PARAMS}" \\
    --levels="${LEVELS}" \\
    ${OVERWRITE}
EOT
