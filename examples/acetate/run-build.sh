#!/bin/bash

#SBATCH --job-name=build
#SBATCH --constraint=gen-a
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=10G
#SBATCH --time=0-03:55

set -euo pipefail
cd "$(dirname "$0")"

source ~/.bashrc
gomamba
mamba activate bff

bff build configs/build-colvars.yaml
