# Desc: Hysia dockerfile with anaconda.
# Author: Zhou Shengsheng
# Date: 28-03-19

ARG CUDA="9.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Copy source
COPY . content

# Set Environment
ENV CONDA_ROOT=/miniconda
ENV PATH=${CONDA_ROOT}/bin:/usr/local/cuda:${PATH}
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# install basics
RUN apt-get update \
 && xargs apt-get install -y < /content/docker/buildpkg.txt \
 && xargs apt-get install -y < /content/docker/runtimepkg.txt \
 && apt-get purge -y nodejs npm \
 && wget https://deb.nodesource.com/setup_8.x -O setup_nodejs.sh \
 && bash setup_nodejs.sh \
 && rm setup_nodejs.sh \
 && apt-get clean \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /var/cache/apt/archives

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh\
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p ${CONDA_ROOT} \
 && rm /miniconda.sh

# Create a Python 3.6 environment
RUN conda env update --name base --file /content/environment.yml \
 && conda clean -ya \
 && rm -rf ~/.cache/pip
RUN (find ${CONDA_ROOT}/ -follow -type f -name '*.a' -delete 2> /dev/null || :) \
 && (find ${CONDA_ROOT}/ -follow -type f -name '*.pyc' -delete 2> /dev/null || :) \
 && (find ${CONDA_ROOT}/ -follow -type f -name '*.js.map' -delete 2> /dev/null || :) \
 && (find ${CONDA_ROOT}/lib/python*/site-packages/bokeh/server/static \
     -follow -type f -name '*.js' ! -name '*.min.js' -delete 2> /dev/null || :)

# Modify the environment variables
#ENV PATH /usr/local/cuda/bin:${HOME}/anaconda3/bin:${PATH}
#ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR content
# Build hysia
RUN bash scripts/build.sh

# Uninstall build dependency
RUN xargs apt-get remove -y < /content/docker/buildpkg.txt \
 && apt-get clean \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash", "/content/scripts/docker_entry.sh"]
