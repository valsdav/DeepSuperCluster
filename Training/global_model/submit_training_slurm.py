import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--basedir", type=str, help="Base dir", default=os.getcwd())
parser.add_argument("--name", type=str, help="Model version name", required=False, default="base")
parser.add_argument("--config", type=str, help="config file (relative to base dir)", required=True)
parser.add_argument("--model", type=str, help="Model.py (relative to basedir)", required=True)
parser.add_argument("--test", action="store_true", help="Do no run condor job but interactively")
args = parser.parse_args()

# Checking the input files exists
os.makedirs(args.basedir, exist_ok=True)
os.makedirs(f"{args.basedir}/condor_logs", exist_ok=True)

if not os.path.exists(os.path.join(args.basedir, args.config)):
    raise ValueError(f"Config file does not exists: {args.config}")
if not os.path.exists(os.path.join(args.basedir, args.model)):
    raise ValueError(f"Model file does not exists: {args.model}")

if args.test:
    os.system(f"sh run_training_slurm.sh {args.basedir}/{args.config} {args.basedir}/{args.model} {args.name}")
    exit(0)

    
slurm_script = f'''#!/bin/bash

#SBATCH -J {args.name}
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu
#SBATCH --nodes=1                        # request to run job on single node
#SBATCH --ntasks=5                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job
#SBATCH --mem=20G                        # memory (per job)
#SBATCH --time=03-00:00:00
#SBATCH --gres-flags=disable-binding

__conda_setup="$('/work/dvalsecc/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work/dvalsecc/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/work/dvalsecc/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work/dvalsecc/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup


#Activate the env
conda activate tfgpu

#Activation cuda libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export PATH=$CONDA_PREFIX/bin/:$PATH


cd /work/dvalsecc/Clustering/DeepSuperCluster/Training/global_model

python trainer_awk.py --config {args.config} --model {args.model} --name {args.name}

'''

with open(f"sbatch_{args.name}", "w") as o:
    o.write(slurm_script)
