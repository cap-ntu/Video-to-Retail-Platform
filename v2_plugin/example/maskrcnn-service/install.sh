#!/usr/bin/env bash

# install basics
apt-get update -y \
  && xargs apt-get install -y < buildpkg.txt \
  && xargs apt-get install -y < runtimepkg.txt \
  && apt-get clean

mkdir -p third && cd third || exit 1
THIRD_DIR=${PWD}

# install PyTorch ${CUDA_VERSION}
conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch -y

# install apex
git clone https://github.com/NVIDIA/apex.git && cd apex || exit 1
python setup.py install --cuda_ext --cpp_ext

# install maskrcnn-benchmark
cd "${THIRD_DIR}" || exit 1
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && python setup.py build develop

# uninstall build dependency
xargs apt-get remove -y < buildpkg.txt \
 && apt-get clean \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# make symbolic link
cd "${THIRD_DIR}"/.. || exit 1
#ln -s "${PWD}"/third/maskrcnn-benchmark "${PWD}"/inference
cp third/maskrcnn-benchmark/demo/predictor.py mask_rcnn_predictor.py
