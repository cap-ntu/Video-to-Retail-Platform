# Deploy Hysia via Docker

## Build Docker Image

Enter the root directory of Hysia and run the following command:
```bash
docker build -t hysia -f docker/hysia-conda-Dockerfile .
```

This build will use `Anaconda` to manage `Hysia`'s dependencies.
We also provided `pip` version, however, with `pip` version the `search` module cannot be used.
Run the following command if you prefer the `pip` version:
```bash
docker build -t hysia -f docker/hysia-pip-Dockerfile .
```

The building process is quite slow as it will install all of the dependencies.
Once built, a docker image with tag `hysia:latest` will be generated.

Optionally use the following command to check if Hysia image is built successfully:
```bash
docker images | grep hysia:latest
```

## Deployment

To deploy Hysia via docker, you must build the docker image first. To deploy, run the following command:
```bash
docker run --runtime=nvidia -d -p 8000:8000 hysia:latest
```

Optionally use the following command to check if Hysia is running correctly:
```bash
docker ps | grep hysia:latest
```

To visit the deployed Hysia dashboard, visit the following address:
```
http://your-server-ip:8000
```
