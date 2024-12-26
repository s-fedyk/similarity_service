# Start from a lightweight Python base image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p /root/.deepface/weights && chmod -R 777 /root/.deepface

# Copy just your requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your project into /app
COPY . .

# Expose port 50051 if you're running a gRPC server on that port
EXPOSE 50051

# Command to run your Python application
CMD ["python", "main.py"]
