# Instructions

## Setting enviroment
To run the model training/inference you need to setup all the required configuration for accelerate and wandb.
First of all, you need to load the required modules on Leonardo:


source $HOME/Conda_init.txt

module load profile/deeplrn

module load cuda/11.8

module load gcc/11.3.0

module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8  

module load llvm/13.0.1--gcc--11.3.0-cuda-11.8

module load nccl/2.14.3-1--gcc--11.3.0-cuda-11.8

module load gsl/2.7.1--gcc--11.3.0-omp

module load fftw/3.3.10--gcc--11.3.0

After that you can load the conda environment:
conda activate /leonardo/pub/userexternal/sdigioia/sdigioia/env/RLenv

Then you can generate the default config for accelerate:
accelerate config

This config should be locate where is the ACCELERATE CACHE directory.

In parallel you need to create a wandb account, and add to the config (for preprocessing, training or inference)
your wandb API key and user name

For the training of the model

bash run.sh config/train_Rall
