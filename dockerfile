# --------------------------------------------------
# Stage 1: Build Python 3.12.3 from source
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
# Stage 2: Final runtime image
# --------------------------------------------------
FROM amazonlinux:2023

# Copy Python 3.12.3 binaries from the builder stage
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

# Example: install your requirements
COPY requirements.txt .
RUN pip3.10 install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Example compilation of your .proto files (if needed)
RUN python3.10 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/ImageService.proto

EXPOSE 50051

CMD ["python3.10", "main.py"]
