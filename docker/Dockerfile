# Desc: Hysia dockerfile with anaconda.
# Author: Zhou Shengsheng
# Date: 28-03-19

FROM hysia/hysia:base

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Copy source
RUN rm -rf content
COPY . content

# Set Environment
ENV CONDA_ROOT=/miniconda
ENV PATH=${CONDA_ROOT}/bin:/usr/local/cuda:${PATH}
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
ENV PYTHONPATH=$PYTHONPATH:/content/hysia
ENV CPU_ONLY=1

WORKDIR content
# Build hysia
RUN bash scripts/build.sh

# Uninstall build dependency
RUN xargs apt-get remove -y < /content/docker/buildpkg.txt \
 && apt-get clean \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash", "/content/scripts/docker_entry.sh"]
