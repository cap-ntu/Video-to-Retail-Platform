# Clipper Baseline

## Prerequisites

- Clipper relies on docker to support its containerization, so make sure `docker` and `nvidia-docker` are installed.
- Clipper doesn't provide any method to specify runtime for a container. It is necessary to change docker's default runtime to `nvidia` globally.
Modify `/etc/docker/daemon.json` to add the following changes:
    ```
    "default-runtime":"nvidia"
    ```
    And then restart docker with:
    ```bash
    service restart docker
    ```
- Install python dependencies that global configuration file doesn't have:
    - soundfile
    - requests_futures

## Build customized clipper model image

In order to get hysia models running in clipper, the model container must be customized. The reasons are given below:

- Clipper use `cloudpickle` to serialize python functions.
However, only codes within the single source file can be serialized.
Dependencies (including third party's and our own) written as `import ...` won't be serialized.
That's why an `ImportError` will be raised if one tries to deploy model with dependencies to clipper.
- Clipper provides some predefined model containers for running `TensorFlow`, `PyTorch`, `MXNet` and other models.
But one must provides both network structures and weights. And all pre-processing and post-processing must be done with the single source file.
- Clipper doesn't support input data that is high dimensional. Because clipper uses json while doesn't provide necessary serialization mechanism.

Use `build-hysia-clipper-base-container-gpu.sh` to build customized clipper model image as base image.
The script must be run in the root directory of hysia project as it copies hysia codes and dependencies like `hysia`, `third` and `weights` to container:

```bash
sh hysia/baseline/clipper/custom-model-container/build-hysia-clipper-base-container-gpu.sh
```

**Note**: 
- The customized image only needs to be built once. All the model containers will use it as their base image.
But if the environment of hysia changes, it is needed to rebuild the image. 
- The building process will be quiet long because it needs to install all the necessary binary and python 
dependencies in the container and copy cuda and hysia codes from localhost to container.

## Deploy model

To deploy a model to clipper, move to the `hysia/baseline/clipper` directory and run `deploy_clipper_model.py` with `model_name` and `model version` parameters:
```bash
cd hysia/baseline/clipper
python deploy_clipper_model.py --model_name=mmdet --model_version=1
```

**Note**:
- Each model only needs to be deployed once. 
- The deployment script will send customized `python_closure_container.py` to the container to override the original one.
This is to bypass clipper's own way to deserialize functions (i.e. through `cloudpickle`) so that functions can be loaded natively.
And this also eliminates the disability to handle high dimensional input data by writing our own serialization methods. 
- Supported model: `mmdet`, `imagenet`.

This will create a new model container to serve mmdet and copy customized `python_closure_container.py` to replace the original one in the container.

## Test model inference throughput

To test clipper model inference throughput, move to the `tests/test_clipper` directory and run `test_clipper_throughput.py` script with `model_name` and other parameters:
```bash
cd tests/test_clipper
python test_clipper_throughput.py --model_name=mmdet --num_tests=100 --concurrency=8
```

**Note**: 
- It might be necessary to wait for a while after you have deployed a model to clipper in order to test it even if you found that model container is running and logging without error.
Clipper may need some time to setup the model container (or may have some bugs here). 
- Supported model: `mmdet`, `imagenet`.
