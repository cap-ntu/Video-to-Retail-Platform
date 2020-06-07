#!/usr/bin/env bash

BASE_DIR=${PWD}

echo "compile decode module"
cd "${BASE_DIR}"/hysia/core/HysiaDecode || return 1
make clean

echo "obtain nv driver version"
# TODO(lym): Get NV driver version for docker build, one solution could be rebuild when GPU is enabled in runtime.
if [[ "$BUILD_FLAG" == "1" ]] ; then
  make CPU_ONLY=TRUE
else
  major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1 | cut -d. -f1)
  # check if nv driver major version higher than 396
  if ((major > 396))
  then
    make
  else
    make CPU_ONLY=TRUE
  fi
fi

# build mmdect
echo "Building mmdect"
cd "${BASE_DIR}"/third || return 1
bash ./compile.sh

# build server
echo "Building server"

cd "${BASE_DIR}"/v2_plugin || return 1
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. protos/infer.proto

cd "${BASE_DIR}"/server || return 1
# generate rpc
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. protos/api2msl.proto

export HYSIA_BUILD=TRUE
bash ./reset-db.sh
unset HYSIA_BUILD
