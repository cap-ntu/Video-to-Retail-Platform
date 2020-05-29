# Contributing

Thank you for interest in contributing to Hysia!

**V2** is under active development currently. You are welcome to create a issue, pull request here. We will credit them
into V2.

## Setup Environment

Setup the developing environment by downloading data and install environments locally. All script is running from the 
project root.

### 1. Download Data

Here is a summary of required data / packed libraries.

| File name     | Description | File ID | Unzipped directory  |
| ------------- | ----------- | ------- | ------------------- |
| [hysia-decoder-lib-linux-x86-64.tar.gz] | Hysia Decoder dependent lib | 1fi-MSLLsJ4ALeoIP4ZjUQv9DODc1Ha6O | `hysia/core/HysiaDecode` |
| [weights.tar.gz] | Pretrained model weights | 1O1-QT8HJRL1hHfkRqprIw24ahiEMkfrX | `.` |
| [object-detection-data.tar.gz] | Object detection data | 1an7KGVer6WC3Xt2yUTATCznVyoSZSlJG | `third/object_detection` |

For users without Google Drive access, you can download from [Baidu Wangpan](https://pan.baidu.com/s/12ZsA__TSNPl0riQ6hSciFQ) and unzip files correspondingly.  

[hysia-decoder-lib-linux-x86-64.tar.gz]: https://drive.google.com/open?id=1fi-MSLLsJ4ALeoIP4ZjUQv9DODc1Ha6O
[weights.tar.gz]: https://drive.google.com/file/d/1O1-QT8HJRL1hHfkRqprIw24ahiEMkfrX/view?usp=sharing
[object-detection-data.tar.gz]: https://drive.google.com/file/d/1an7KGVer6WC3Xt2yUTATCznVyoSZSlJG/view?usp=sharing

#### Option 1: auto-download

```shell script
bash scripts/download-data.sh
```

#### Option 2: manually download

1\. Download [Hysia Decoder dependent libraries](https://drive.google.com/file/d/1O1ewejZbMWj43IxL7NInuJss7fNjYc3R) and unzip it:
```shell script
deocder_path=hysia/core/HysiaDecode
mv hysia-decoder-lib-linux-x86-64.tar.gz "${deocder_path}"
cd "${deocder_path}"
tar xvzf hysia-decoder-lib-linux-x86-64.tar.gz
rm -f hysia-decoder-lib-linux-x86-64.tar.gz
cd -
```

2\. Download pretrained [model weights](https://drive.google.com/file/d/1O1-QT8HJRL1hHfkRqprIw24ahiEMkfrX/view?usp=sharing)
and unzip it:
```shell script
tar xvzf weights.tar.gz
# and remove the weights zip
rm -f weights.tar.gz
```

3\. Download [object detection data](https://drive.google.com/file/d/1an7KGVer6WC3Xt2yUTATCznVyoSZSlJG/view?usp=sharing)
in third-party library and unzip it:
```shell script
mv object-detection-data.tar.gz third/object_detection
cd third/object_detection
tar xvzf object-detction-data.tar.gz
rm object-detection-data.tar.gz
cd -
```

### 2. Installation

These scripts are tested on Ubuntu 16.04 x86-64 with CUDA9.0 and CUDNN7. Docker is recommended for other system 
version.

Requirements:
- Conda
- Nvidia driver
- CUDA = 9
- CUDNN
- g++
- zlib1g-dev

#### Option 1: auto-installation 

Run the following script:
```shell script
# Execute this script at project root
bash scripts/install-build.sh
```

#### Option 2: step-by-step installation

```shell script
# Firstly, make sure that your Conda is setup correctly and have CUDA,
# CUDNN installed on your system.

# Install Conda virtual environment
conda env create -f environment.yml

conda activate V2O

export BASE_DIR=${PWD}

# Compile HysiaDecoder
cd "${BASE_DIR}"/hysia/core/HysiaDecode
make clean
# If nvidia driver is higher than 396, set NV_VERSION=<your nvidia major version>
make NV_VERSION=<your nvidia driver major version>

# Build mmdetect
# ROI align op
cd "${BASE_DIR}"/third/
cd mmdet/ops/roi_align
rm -rf build
python setup.py build_ext --inplace

# ROI pool op
cd ../roi_pool
rm -rf build
python setup.py build_ext --inplace

# NMS op
cd ../nms
make clean
make PYTHON=python

# Initialize Django
# This will prompt some input from you
cd "${BASE_DIR}"/server
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. protos/api2msl.proto

python manage.py makemigrations restapi
python manage.py migrate
python manage.py loaddata dlmodels.json
python manage.py createsuperuser

unset BASE_DIR
```

### (Optional) Rebuild the frontend
If the frontend is updated, please run the following:  

#### Option 1: auto-build

```shell script
cd server/react-build
bash ./build.sh
cd ..
```

#### Option 2: manually build

```shell script
cd server/react-front

# Install dependencies
npm i
npm audit fix

# Build static files
npm run-script build

# fix js path
python fix_js_path.py build

# create a copy of build static files
mkdir -p tmp
cp -r build/* tmp/

# move static folder to static common
mv tmp/*html ../templates/
mv tmp/* ../static/
cp -rfl ../static/static/* ../static/
rm -r ../static/static/

# clear temp
rm -r tmp
```

## Testing

## Demo

Start model server:
```shell script
cd server

# Start model server
python start_model_servers.py
```

Start Django server
```shell script
cd server

# Run Django
python manage.py runserver 0.0.0.0:8000
```
