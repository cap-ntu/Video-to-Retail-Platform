#!/usr/bin/env bash
# Desc: Script to build hysia-clipper-base-container-gpu-Dockerfile.
# Author: Zhou Shengsheng
# Date: 15-02-19

set -e

DOCKERFILE="hysia/baseline/clipper/custom-model-container/hysia-clipper-base-container-gpu-Dockerfile"
CLIPPER_VERSION="0.3.0"
CLIPPER_URL="https://github.com/ucbrise/clipper/archive/v${CLIPPER_VERSION}.tar.gz"

if [[ ! -f ${DOCKERFILE} ]]; then
    echo "This script must be run in the root directory of hysia project!"
    echo 'Run this script as "sh hysia/baseline/clipper/custom-model-container/build-hysia-clipper-base-container-gpu.sh"'
    exit 1
fi

# Download clipper source code
echo "Downloading clipper source code"
wget ${CLIPPER_URL} -O clipper.tar.gz
tar -xf clipper.tar.gz
mv clipper-${CLIPPER_VERSION} clipper

# Copy cuda to working dir
# cp -r /usr/local/cuda-9.0 .

# Build docker image
echo "Building docker image with hysia-clipper-base-container-gpu-Dockerfile"
docker build -t hysia-clipper-base-container-gpu -f ${DOCKERFILE} .

# Cleanup
echo "Cleaning up"
rm -rf clipper.tar.gz
rm -rf clipper
rm -rf cuda-9.0

echo "Done!"
