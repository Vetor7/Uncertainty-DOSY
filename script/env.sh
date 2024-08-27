#!/bin/bash

set -e

anaconda_dir=/root/anaconda3 # Your conda dir

. $anaconda_dir'/etc/profile.d/conda.sh'
conda remove -n DOSY --all -y
conda create -n DOSY python=3.10 -y
conda activate DOSY
conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt