#!/bin/bash
 
#SBATCH --job-name=comp                           # Job name, will show up in squeue output
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20                        # Number of cores
#SBATCH --nodes=1                                 # Ensure that all cores are on one machine
#SBATCH --time=0-00:10:00                         # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=1000                        # Memory per cpu in MB (see also --mem) 
#SBATCH --mail-type=END                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=steven.thomson@fu-berlin.de   # Email to which notifications will be sent 

# Run script
bash
#conda install cython
#python cudatest.py
#cat /proc/driver/nvidia/version
#nvidia-smi -L
#python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"

#python main_itc.py 8 random einsum ${SLURM_ARRAY_TASK_ID}
python setup.py build_ext --inplace
