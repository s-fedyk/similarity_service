FROM nvidia/cuda:12.6.3-runtime-amzn2023

RUN dnf install -y python3 python3-pip mesa-libGL
 
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.deepface/weights && chmod -R 777 /root/.deepface

COPY . .

RUN python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/ImageService.proto

EXPOSE 50051

CMD ["python3", "main.py"]
