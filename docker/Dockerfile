 
ARG BASE_DOCKER="nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04"
FROM ${BASE_DOCKER}
# Args before FROM have to be declaired again after FROM
ARG BASE_DOCKER

# Avoid user dialogs
ENV DEBIAN_FRONTEND=noninteractive
ARG CI_COMMIT_REF_NAME=xxxxxx
ARG CI_JOB_TOKEN=xxxxxx

# choose requirementsfile for install
ARG REQUIREMENT_FILES="base.txt pt110.txt yolox.txt"
ARG ENTRYPOINT_FILE="entrypoint_docker.sh"

# create Build directory - will be deleted at the end 
ENV BUILD_HOME /tmp/tmp_build
RUN mkdir ${BUILD_HOME}


# Essential tools
RUN apt-get update -q && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake \
    libopenblas-dev \
    locales \
    && rm -rf /var/lib/apt/lists/*

# install sudo
RUN apt-get update -q && apt-get install -y sudo --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Python 3.8
RUN apt-get update -q && apt-get install -y --no-install-recommends python3.8 python3.8-dev python3-pip python3-tk \
   && rm -rf /var/lib/apt/lists/*
# Set python3.8 as default python version
RUN rm /usr/bin/python3 \
    && ln -s python3.8 /usr/bin/python3
# Install pip and correct version of setuptools for ml_core
RUN cd $BUILD_HOME \ 
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py --force-reinstall 

# OpenCV dependencies
RUN apt-get update -q \
    && apt-get install -y libgtk2.0-dev --no-install-recommends \
    && apt-get install -y libgl1-mesa-glx --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# copy all requirements files
COPY .docker/requirements/ ${BUILD_HOME}/
RUN ls -la ${BUILD_HOME}
# install all requirements files defined in ${REQUIREMENT_FILES}
RUN cd ${BUILD_HOME} \
    && for R_FILE in ${REQUIREMENT_FILES}; do pip3 install -r ${R_FILE}; done

# Install ml_core - develop
# To build the docker LOCALLY
# copy/clone ml_core to .docker
# RUN mkdir /home/ml_core
# COPY .docker/ml_core /home/ml_core
# RUN cd /home/ml_core  \
#     && pip3 uninstall -y ml_core \
#     && python3 setup.py install --without-onnxruntime  
#     # onnxruntime-gpu will be installed in the requirements file 
RUN cd /home/ \
    && git clone --branch develop https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.virtualbaker.com/MachineLearning/ml_core.git \
    && cd ml_core \
    && pip3 uninstall -y ml_core \
    && python3 setup.py install --without-onnxruntime  
    # onnxruntime-gpu will be installed in the requirements file 

# install git lfs
RUN mkdir ${BUILD_HOME}/git_lfs \
    && cd ${BUILD_HOME}/git_lfs \
    && wget https://github.com/git-lfs/git-lfs/releases/download/v2.5.2/git-lfs-linux-amd64-v2.5.2.tar.gz \
    && tar -xvf git-lfs-linux-amd64-v2.5.2.tar.gz \
    && sudo ./install.sh \
    && git lfs install

USER root
COPY .docker/${ENTRYPOINT_FILE} /entrypoint.sh
RUN chmod a+rwX /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# remove all build files
RUN rm -rf ${BUILD_HOME}

# create deep_learner diretory as work dir
# this will be also be the location where
# the deep_learner will be mounted
RUN mkdir /home/deep_learner
WORKDIR /home/deep_learner
