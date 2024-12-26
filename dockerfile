FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.deepface/weights && chmod -R 777 /root/.deepface

COPY . .

EXPOSE 50051

# Command to run your Python application
CMD ["python", "main.py"]
