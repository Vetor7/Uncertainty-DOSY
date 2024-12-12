#!/bin/bash

set -e

anaconda_dir=~/anaconda3 # Your conda dir

. $anaconda_dir'/etc/profile.d/conda.sh'
conda remove -n DOSY --all -y
conda create -n DOSY python=3.11 -y
conda activate DOSY

pip install -r requirements.txt