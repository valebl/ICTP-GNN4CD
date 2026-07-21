## GNN4CD-CORDEXML - branch Public

This folder contains the public code of the GNN4CD emulator, updated to its most recent stable version.

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
4. Activate the RLenv environment
`conda activate /leonardo/pub/userexternal/sdigioia/sdigioia/env/RLenv`

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

