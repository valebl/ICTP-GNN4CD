# GNN4CD-CORDEXML - branch Public

This folder contains the public code of the GNN4CD emulator, updated to its most recent stable version.

# Instructions

### Clone the `Public` branch

`git clone -b Public https://github.com/valebl/ICTP-GNN4CD.git`

### Set-up on Leonardo
To set up the environment on Leonardo you need to follow the steps listed below:

1. Move to the cloned directory
`cd ICTP-GNN4CD`
2. Initialize conda
`source conda_init`
3. Load the necessary modules
`source load_modules`
4. Activate the GNN4CDenv environment
`conda activate /leonardo/pub/userexternal/vblasone/envs/GNN4CDenv`

### Create a wandb account
To train the emulator you should create a wandb account on the [wandb website](https://wandb.ai/site/).
On your profile, you need to look for your `USERNAME` and `API key`.

### Create an accelerate config file
Move to your `$HOME` directory and run the accelerate configuration routine
`accelerate config`
You will be asked some questions about your training desired settings and it will create a config file. If you want to train in a multi-GPU setting you should choose `multi-GPU` and specify the desired number (>1) of GPUs.
`~/.cache/huggingface/accelerate/default_config.yaml`
Please rename this file adding a subscript with the number of GPUs specified in the file generation, e.g. if you chose 4 GPUs:
`mv ~/.cache/huggingface/accelerate/default_config.yaml ~/.cache/huggingface/accelerate/default_config_4.yaml`

You should now be all set up to start using the GNN4CD emulator!
Move back to the ICTP-GNN4CD folder and create a new `config` directory. Here you will save your own configuration files to preprocess/train/predict!

# A simple working example

In this example we will see how to run the preprocessing the CORDEX-ML Bench data for the ALPS domain in the ESD pseudo reality training setting.
Then we will train a simple GNN4CD emulator on tasmax for the period 1961-1979 and see the training statistics on wandb.
Finally we will create the predictions for the year 1980 and produce a sample PDF report to evaluate the emulator's predictions against the ground truth.

First, copy the config_default directory into your config folder. We will then modify these files.

`cp -r config_template/ config/`

### Preprocessing the data

You need to fill the following parameters in the config/preprocess_ALPS_esd bash file:

- `LOG_PATH` the complete path to where you want to save the pre-processed data
- `MAIN_PATH` the complete path to your `ICTP-GNN4CD` folder

All the other parameters can be customised but will not address it in this example.

To run the preprocessing:

`./scripts/run_preprocess.sh config/preprocess/preprocess_ALPS_esd`

You can check the state of your job using (substitute your_username with your actual username):

`squeue -u your_username`

The output of your job will be saved in `LOG_PATH` and will consist of the following files:

- `low_high_graph_edgeattr.pkl`
- `low_input.npy`
- `target_tasmax.npy`
- `target_pr.npy`
- `orog.npy`
- `mask_sealand.npy`
- `coords_ij.npy`
- `unique_src.npy`
- `time_index.npy`
- `high_time_index.npy`
- `low_input_metadata.json`
- `target_metadata.json`
- `low2high_norm_constants.json`
- `high_norm_constants.json`

### Training the emulator
You need to fill the following parameters in the config/train_ALPS_esd_tasmax bash file:

- `LOG_PATH` the complete path to where you want to save the pre-processed data
- `WANDB_API_KEY` your wandb API key
- `WANDB_USERNAME` your wandb username
- `MAIN_PATH` the complete path to your `ICTP-GNN4CD` folder
- `INPUT_PATH` the path where you saved the preprocessed data

All the other parameters can be customised but will not address it in this example.

To run the training:

`./scripts/run_train.sh config/train/train_ALPS_esd_tasmax`

You can check the state of your job using (substitute your_username with your actual username):

`squeue -u your_username`

The output of your job will be saved in `LOG_PATH`.

Wandb is set to run offline as the compute nodes on Leonardo do not have internet access. Whenever you want (during training or after training) you can synchronise the wandb logs to the web-platform by running the following command:

`wandb sync --sync-all`
This will upload all your local runs, saved in `ICTP-GNN4CD/wandb`

### Using the trained emulator for predictions
You need to fill the following parameters in the config/predict_ALPS_esd_tasmax bash file:

- `LOG_PATH` the complete path to where you want to save the predictions results and plot report
- `TRAIN_PATH` the path where you saved the trainining output
- `EPOCH` the epoch of the training that you want to use for the predictions (e.g. 149, the last epoch)
- `MAIN_PATH` the complete path to your `ICTP-GNN4CD` folder
- `INPUT_PATH` the path where you saved the preprocessed data

All the other parameters can be customised but will not address it in this example.

To run the predictions:

./scripts/run_predict.sh config/predict/predict_ALPS_esd_tasmax

You can check the state of your job using (substitute your_username with your actual username):

`squeue -u your_username`

The output of your job will be saved in `LOG_PATH`. The output of the predictons are a pickle file containing the graph structure and target/prediction data. Also, a PDF report with some comparison plots is automatically created.

