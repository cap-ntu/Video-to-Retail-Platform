#!/usr/bin/env bash

# install Conda virtual environment
conda env create -f environment.yml

eval "$(conda shell.bash hook)"
conda activate Hysia

# call build
cd ..
bash scripts/build.sh
