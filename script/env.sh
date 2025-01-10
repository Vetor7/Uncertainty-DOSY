#!/bin/bash

set -e

# Define your Anaconda directory
anaconda_dir=~/anaconda3 # Update if necessary

# Source Conda
if [ -f "$anaconda_dir/etc/profile.d/conda.sh" ]; then
  . "$anaconda_dir/etc/profile.d/conda.sh"
else
  echo "Error: conda.sh not found. Check your Anaconda installation path."
  exit 1
fi

# Remove and recreate the environment
conda remove -n DOSY --all -y
conda create -n DOSY python=3.11 -y
conda activate DOSY

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "Error: requirements.txt not found."
  exit 1
fi

echo "Environment setup complete."
