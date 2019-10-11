# Deploy Hysia via Docker

## Build Docker Image

Make sure your machine has installed Nvidia Docker.  
Enter the root directory of Hysia and run the following command:
```shell script
docker build -t hysia -f docker/Dockerfile .
```

Once built, a docker image with tag `hysia:latest` will be generated.

Optionally use the following command to check if Hysia image is built successfully:
```shell script
docker images | grep hysia:latest
```

## Deployment

To deploy Hysia via docker, you must build the docker image first. To deploy, run the following command:
```shell script
docker run --gpus all -d -p 8000:8000 hysia:latest
```

Optionally use the following command to check if Hysia is running correctly:
```shell script
docker ps | grep hysia:latest
```

Visit the following address and use `username=admin`, `password=admin` for login: http://locahost:8000
