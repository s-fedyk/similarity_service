FROM ubuntu:20.04

LABEL maintainer="Amazon AI"
#SDK 1.17.1 has version 1. We skipped 1.18.0.
LABEL dlc_major_version="1"
# Specify accept-bind-to-port LABEL for inference pipelines to use SAGEMAKER_BIND_TO_PORT
# https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipeline-real-time.html
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ARG PYTHON=python3.10
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.10.12
ARG TFS_SHORT_VERSION=2.10

# Neuron SDK components version numbers
ARG NEURONX_RUNTIME_LIB_VERSION=2.16.*
ARG NEURONX_COLLECTIVES_LIB_VERSION=2.16.*
ARG NEURONX_TOOLS_VERSION=2.13.*
ARG NEURONX_FRAMEWORK_VERSION=2.10.1.2.1.*
ARG NEURONX_TF_MODEL_SERVER_VERSION=2.10.1.2.10.1.*
ARG NEURONX_CC_VERSION=2.9.*

# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8
# Python won’t try to write .pyc or .pyo files on the import of source modules
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SAGEMAKER_TFS_VERSION="${TFS_SHORT_VERSION}"
ENV PATH="/opt/aws/neuron/bin:$PATH:/sagemaker"
ENV LD_LIBRARY_PATH='/opt/aws/neuron/lib:/usr/local/lib:$LD_LIBRARY_PATH'
ENV MODEL_BASE_PATH=/models
# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=model
ENV DEBIAN_FRONTEND=noninteractive

# nginx + njs
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get -y upgrade --only-upgrade systemd \
 && apt-get -y install --no-install-recommends \
    curl \
    gnupg2 \
    ca-certificates \
    emacs \
    git \
    unzip \
    wget \
    vim \
    libbz2-dev \
    liblzma-dev \
    libffi-dev \
    build-essential \
    zlib1g-dev \
    openssl \
    libssl1.1 \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libcap-dev \
    libc6-dev \
 && curl -s http://nginx.org/keys/nginx_signing.key | apt-key add - \
 && echo 'deb http://nginx.org/packages/ubuntu/ focal nginx' >> /etc/apt/sources.list \
 && apt-get update \
 && apt-get -y install --no-install-recommends \
    nginx=1.20.1* \
    nginx-module-njs=1.20.1* \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install python3.10
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz \
 && tar -xvf Python-$PYTHON_VERSION.tgz \
 && cd Python-$PYTHON_VERSION \
 && ./configure && make && make install \
 && rm -rf ../Python-$PYTHON_VERSION* \
 && rm -rf /tmp/tmp*

RUN echo "deb https://apt.repos.neuron.amazonaws.com focal main" > /etc/apt/sources.list.d/neuron.list
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -
RUN apt-get update

RUN apt-get install -y \
    tensorflow-model-server-neuronx=${NEURONX_TF_MODEL_SERVER_VERSION} \
    aws-neuronx-tools=${NEURONX_TOOLS_VERSION} \
    aws-neuronx-collectives=${NEURONX_COLLECTIVES_LIB_VERSION} \
    aws-neuronx-runtime-lib=${NEURONX_RUNTIME_LIB_VERSION} \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# cython, falcon, gunicorn, grpc
RUN ${PIP} install --no-cache-dir \
    "awscli<2" \
    boto3 \
    cython==0.29.* \
    falcon==2.* \
    gunicorn==20.1.* \
    gevent==21.12.* \
    requests \
    grpcio==1.56.0 \
    "protobuf<4" \
# using --no-dependencies to avoid installing tensorflow binary
 && ${PIP} install --no-dependencies --no-cache-dir \
    tensorflow-serving-api==2.10.1

# pip install statements have been separated out into multiple sequentially executed statements to
# resolve package dependencies during installation.
RUN ${PIP} install neuronx-cc==${NEURONX_CC_VERSION} tensorflow-neuronx==${NEURONX_FRAMEWORK_VERSION} --extra-index-url https://pip.repos.neuron.amazonaws.com \
 && ${PIP} install tensorboard-plugin-neuron --extra-index-url https://pip.repos.neuron.amazonaws.com

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python \
 && ln -s $(which ${PIP}) /usr/bin/pip

RUN curl https://tensorflow-aws.s3-us-west-2.amazonaws.com/MKL-Libraries/libiomp5.so -o /usr/local/lib/libiomp5.so
RUN curl https://tensorflow-aws.s3-us-west-2.amazonaws.com/MKL-Libraries/libmklml_intel.so -o /usr/local/lib/libmklml_intel.so

RUN HOME_DIR=/root \
 && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
 && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
 && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
 && chmod +x /usr/local/bin/testOSSCompliance \
 && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
 && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} \
 && rm -rf ${HOME_DIR}/oss_compliance*

RUN curl https://aws-dlc-licenses.s3.amazonaws.com/tensorflow-$TFS_SHORT_VERSION/license.txt -o /license.txt

COPY requirements.txt .
RUN ${PIP} freeze > base_requirements.txt
RUN ${PIP} install --no-cache-dir -r requirements.txt -c base_requirements.txt

COPY . .

RUN ${PYTHON} -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/ImageService.proto
RUN ${PYTHON} -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/Analyzer.proto
RUN ${PYTHON} -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/Preprocessor.proto


EXPOSE 50051

ENV XLA_USE_BF16="1"
ENV NEURONCORE_GROUP_SIZES="2,2"

CMD ["python3.10", "main.py"]
