ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

# set built-time arguments
ARG CUDA

# set system environment
ENV CONDA_ROOT=/miniconda
ENV CONDA_PREFIX=${CONDA_ROOT}
ENV PATH=${CONDA_ROOT}/bin:${PATH}
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /content

# Change all files EOF to LF
RUN find /content -type f -exec sed -i -e 's/^M$//' {} \;

# Install basic
RUN apt-get update -y \
 && apt-get install -y curl

# Install Miniconda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -o /miniconda.sh \
 && sh /miniconda.sh -b -p "${CONDA_ROOT}" \
 && rm /miniconda.sh

# Install base environment
RUN conda env update --name base -f /content/base-env.yml

WORKDIR /content

ENTRYPOINT ["sh", "docker-entry.sh"]
