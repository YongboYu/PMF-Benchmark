#!/bin/bash

# Create and activate environment
conda create -n pmf-benchmark python=3.11 -y
source activate pmf-benchmark

# Install remaining packages via pip
pip install -r requirements.txt