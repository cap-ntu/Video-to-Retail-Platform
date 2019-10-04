#!/usr/bin/env bash

echo "Building roi align op..."
cd mmdet/ops/roi_align || return 1
if [ -d "build" ]; then
    rm -r build
fi
python setup.py build_ext --inplace

echo "Building roi pool op..."
cd ../roi_pool || return 1
if [ -d "build" ]; then
    rm -r build
fi
python setup.py build_ext --inplace

echo "Building nms op..."
cd ../nms || return 1
make clean
make PYTHON=python
