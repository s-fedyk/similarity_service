# --------------------------------------------------
# Build Python 3.10.0 from source
# --------------------------------------------------
FROM amazonlinux:2023 AS builder

# Install build dependencies
RUN dnf update -y && dnf install -y \
    gcc \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    zlib-devel \
    make \
    wget \
    tar \
    gzip

# Download and build Python 3.12.3
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar xzf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make altinstall

# --------------------------------------------------
# Final runtime image
# --------------------------------------------------
FROM amazonlinux:2023

COPY --from=builder /usr/local/bin/python3.10 /usr/local/bin/
COPY --from=builder /usr/local/bin/pip3.10 /usr/local/bin/
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10

RUN tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
metadata_expire=0
EOF

RUN rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
RUN yum update -y
RUN yum install aws-neuronx-dkms-2.* -y
RUN yum install aws-neuronx-collectives -y
RUN yum install aws-neuronx-runtime-lib -y
RUN yum install aws-neuronx-tools-2.* -y

RUN dnf install -y mesa-libGL

RUN python3.10 --version

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip3.10 install --no-cache-dir -r requirements.txt

RUN python3.10 -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
RUN python3.10 -m pip install tensorflow-neuron[cc] "protobuf"
RUN python3.10 -m pip install tensorboard-plugin-neuron
RUN python3.10 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/ImageService.proto

COPY . .

EXPOSE 50051

CMD ["python3.10", "main.py"]
