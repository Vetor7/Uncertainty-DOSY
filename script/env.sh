#!/bin/bash

set -e

anaconda_dir=/root/anaconda3 # Your conda dir

. $anaconda_dir'/etc/profile.d/conda.sh'
conda remove -n DOSY --all -y
conda create -n DOSY python=3.11 -y
conda activate DOSY

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt