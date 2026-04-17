import configparser
import subprocess
import os
import sys

config_file = sys.argv[1]

cfgp = configparser.ConfigParser()
cfgp.read(config_file)
cfg = cfgp['CONFIG']

NUM_TESTS = int(cfg["NUM_TESTS"])
VAR = cfg["VAR"]
job_name_list = [f"{cfg['JOB_NAME']}_{ii+1}" for ii in range(NUM_TESTS)]
pred_list = [cfg[f'PREDICTORS_FILE_{ii+1}'] for ii in range(NUM_TESTS)]
input_list = [cfg[f'INPUT_PATH_{ii+1}'] for ii in range(NUM_TESTS)]
log_list = [f"log_test_{ii+1}_{VAR}.txt" for ii in range(NUM_TESTS)]
outputP_list = [cfg[f'OUTPUT_PATH_{ii+1}'] for ii in range(NUM_TESTS)]
outputF_list = [cfg[f'OUTPUT_{ii+1}'] for ii in range(NUM_TESTS)]
period_list = [cfg[f'PERIOD_{ii+1}'] for ii in range(NUM_TESTS)]
val_mode_list = [cfg[f'VAL_MODE_{ii+1}'] for ii in range(NUM_TESTS)]
test_id_list = [cfg[f'TEST_ID_{ii+1}'] for ii in range(NUM_TESTS)]

target_type = cfg['TARGET_TYPE']
experiment = cfg['EXPERIMENT']
accelerate_config = cfg['ACCELERATE_CONFIG_PATH']
stats_mode = cfg['STATS_MODE']
epoch = cfg['EPOCH']
CHECKPOINT_R = f"checkpoints/checkpoint_{epoch}"
model_name = cfg['MODEL_NAME']
model_type = cfg['MODEL_TYPE']
RUN_TYPE = cfg['RUN_TYPE']
graph_file = cfg['LOW_GRAPH_FILE']
training_path = cfg['TRAINING_PATH']
input_path = cfg['INPUT_PATH']
plot_path = cfg['PLOT_PATH']
dataset_name = cfg['DATASET_NAME']
OROG_FILE = cfg["OROG_FILE"]
MASK_SEALAND_FILE = cfg["MASK_SEALAND_FILE"]
COORDS_IJ_FILE = cfg["COORDS_IJ_FILE"]
METADATA_FILE = cfg["METADATA_FILE"]
USE_ACCELERATE = cfg["USE_ACCELERATE"]
DOMAIN = cfg["DOMAIN"]
write_slurm = cfg["write_slurm"]
run_slurm = cfg["run_slurm"]
run_report = cfg["run_report"]

os.makedirs(f"slurm_predictions/{experiment}/{VAR}", exist_ok=True)

for ii in range(len(input_list)):

    # 1. WRITE PREDICTION SLURM FILE
    slurm_file = f"slurm_predictions/{experiment}/{VAR}/run_prediction_{VAR}_{DOMAIN}_test_{ii}_{experiment}.slurm"

    slurm_txt = f"""#!/bin/bash
mkdir -p {outputP_list[ii]}logs/
sbatch << EOT
#!/bin/bash
#SBATCH -A ict26_esp_0
#SBATCH -p boost_usr_prod
#SBATCH --time {cfg['TIME']}
#SBATCH -N 1
#SBATCH --mem={cfg['MEM']}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name={job_name_list[ii]}
#SBATCH -o {outputP_list[ii]}logs/run_test_{ii+1}_{VAR}.out
#SBATCH -e {outputP_list[ii]}logs/run_test_{ii+1}_{VAR}.err

module purge
module load --auto profile/meteo
module load --auto profile/deeplrn
module load gcc
module load cdo/2.1.0--gcc--11.3.0
module load cuda/11.8

source {cfg['SOURCE_PATH']}
conda activate {cfg['ENV_PATH']}
cd {cfg['MAIN_PATH']}

python -m accelerate.commands.launch --config_file "{accelerate_config}" test.py \
    --dataset_name="{dataset_name}" \
    --predictors_filename="{pred_list[ii]}" \
    --input_path_P="{input_list[ii]}" \
    --input_path="{input_path}" \
    --output_path="{outputP_list[ii]}" \
    --train_path="{training_path}" \
    --log_path="{outputP_list[ii]}logs/" \
    --output_file="{outputF_list[ii]}" \
    --period="{period_list[ii]}" \
    --experiment={experiment} \
    --domain={DOMAIN} \
    --log_file="logs/{log_list[ii]}" \
    --graph_file="{graph_file}" \
    --batch_size=1 \
    --stats_mode={stats_mode} \
    --target_type={target_type} \
    --model_name="{model_name}" \
    --model_type="{model_type}" \
    --checkpoint_R="{CHECKPOINT_R}" \
    --mask_sealand_file="{MASK_SEALAND_FILE}" \
    --orog_file="{OROG_FILE}" \
    --coords_ij_file="{COORDS_IJ_FILE}" \
    --metadata_file="{METADATA_FILE}" \
    --run_type="{RUN_TYPE}" \
    "{USE_ACCELERATE}"
EOT
"""

    if write_slurm == "True":
        with open(slurm_file, "w") as f:
            f.write(slurm_txt)

    # 2. SUBMIT PREDICTION JOB
    if run_slurm == "True":
        result = subprocess.run(['bash', slurm_file], capture_output=True, text=True, check=True)
        jobid = result.stdout.strip().split()[-1]  # extract job ID
    else:
        jobid = None

    # 3. WRITE REPORT SLURM FILE
    if run_report == "True" and val_mode_list[ii] in ["val_1", "val_2", "era5"]:

        os.makedirs(plot_path+"logs/", exist_ok=True)
        report_name = f"Test_{ii+1}_{val_mode_list[ii]}_{VAR}"
        report_slurm = f"slurm_predictions/{experiment}/{VAR}/run_report_test_{ii+1}_{val_mode_list[ii]}.slurm"

        report_txt = f"""#!/bin/bash
#SBATCH -A ict26_esp_0
#SBATCH -p boost_usr_prod
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH --job-name=report_{ii+1}
#SBATCH -o {plot_path}logs/run_report_{ii+1}_{VAR}.out
#SBATCH -e {plot_path}logs/run_report_{ii+1}_{VAR}.err

source {cfg['SOURCE_PATH']}
conda activate {cfg['ENV_PATH']}
cd {cfg['MAIN_PATH']}

python ./utils/plotting/plot_report_test.py \
    --input_path="{outputP_list[ii]}" \
    --plot_path="{plot_path}" \
    --val_file="{outputF_list[ii]}" \
    --var="{VAR}" \
    --experiment="{experiment}" \
    --period="{period_list[ii]}" \
    --val_mode="{val_mode_list[ii]}" \
    --report_name="{report_name}" \
    --test_id="{test_id_list[ii]}" \
    --domain="{DOMAIN}"
"""

        with open(report_slurm, "w") as f:
            f.write(report_txt)

        # 4. SUBMIT REPORT WITH DEPENDENCY
        if jobid is not None:
            subprocess.run([
                "sbatch",
                f"--dependency=afterok:{jobid}",
                report_slurm
            ])
        else:
            subprocess.run([
                "sbatch",
                report_slurm
            ])
