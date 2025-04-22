#!/bin/bash

# Source conda
source /opt/anaconda3/etc/profile.d/conda.sh
# eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
# Activate your conda environment (replace with your env name)
conda activate base

# Run your Python script
python /Users/veerasarma/XGboostalgo/trainingmodel.py
python /Users/veerasarma/XGboostalgo/predictsignal.py
