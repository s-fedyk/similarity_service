from deepface.models.facial_recognition.Facenet import os
import grpc
from concurrent import futures
import time

from S3Client import getFromS3, initS3, putToS3
import S3Client
import model
import cv2

from proto import ImageService_pb2
from proto import ImageService_pb2_grpc
from proto import Analyzer_pb2
from proto import Analyzer_pb2_grpc
from proto import Preprocessor_pb2
from proto import Preprocessor_pb2_grpc

# singleton
embedder = None
analyzer = None
preprocessor = None

class ImageServicer(ImageService_pb2_grpc.ImageServiceServicer):
    def Identify(self, request, context):
        print(f"identify-{time.time()} {request}")
        global embedder

        start = time.time()
        encodedImage = getFromS3(request.base_image.url, S3Client.bucket_name)

        embedding,faceArea = embedder.extract_embedding(encodedImage, "Facenet512")

        print("finished extraction...")
        response = ImageService_pb2.IdentifyResponse()
        response.embedding.extend(embedding)

        response.facial_area.w = int(faceArea["w"])
        response.facial_area.h = int(faceArea["h"])
        response.facial_area.x = int(faceArea["x"])
        response.facial_area.y = int(faceArea["y"])

        response.facial_area.left_eye.x = int(faceArea["left_eye"][0])
        response.facial_area.left_eye.y = int(faceArea["left_eye"][1])

        response.facial_area.right_eye.x = int(faceArea["right_eye"][0])
        response.facial_area.right_eye.y = int(faceArea["right_eye"][1])

        end = time.time()
        print(f"Responding with: {response}, took {end-start}")
        return response

class AnalysisServicer(Analyzer_pb2_grpc.AnalyzerServicer):
    def Analyze(self, request, context):
        print(f"identify-{time.time()} {request}")
        global analyzer

        start = time.time()
        encodedImage = getFromS3(request.base_image.url, S3Client.bucket_name)

        analysis = analyzer.analyze_face(encodedImage)

        response = Analyzer_pb2.AnalyzeResponse()

        response.age = str(analysis["age"])
        response.gender = analysis["dominant_gender"]
        response.race = analysis["dominant_race"]
        response.emotion = analysis["dominant_emotion"]

        end = time.time()

        print(f"Responding with: {response}, took {end-start}")
        return response

class PreprocessorServicer(Preprocessor_pb2_grpc.PreprocessorServicer):
    def Preprocess(self, request, context):
        print(f"Preprocess-{time.time()} {request}")
        global analyzer
        
        start = time.time()

        encodedImage = getFromS3(request.base_image.url, S3Client.bucket_name)

        img_scaled, scale_w, scale_h, top_pad, left_pad = preprocessor.preprocess(encodedImage)
        print(f"Scales are w:{scale_w}, h:{scale_h}")

        ret, jpeg_buf = cv2.imencode('.jpg', img_scaled)
        if not ret:
            raise ValueError("Failed to encode image to JPEG")

        new_url = f"{request.base_image.url}-processed"

        jpeg_bytes = jpeg_buf.tobytes()
        putToS3(jpeg_bytes, new_url, S3Client.bucket_name)

        response = Preprocessor_pb2.PreprocessResponse()

        response.processed_image.url = new_url
        response.scale_w = scale_w
        response.scale_h = scale_h
        response.top_pad = top_pad
        response.left_pad = left_pad
        
        end = time.time()
        print(f"Responding with: {response}, took {end-start}")

        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    modelType = os.getenv("MODEL_TYPE")
    validTypes = ["embedder", "analyzer", "preprocessor"]
    assert(modelType in validTypes)
    initS3()

    if modelType == "embedder":
        ImageService_pb2_grpc.add_ImageServiceServicer_to_server(ImageServicer(), server)
        global embedder 
        embedder = model.ImageClassifier()
    elif modelType == "analyzer":
        Analyzer_pb2_grpc.add_AnalyzerServicer_to_server(AnalysisServicer(), server)
        global analyzer
        analyzer = model.FaceAnalyzer()
    elif modelType == "preprocessor":
        Preprocessor_pb2_grpc.add_PreprocessorServicer_to_server(PreprocessorServicer(), server)
        global preprocessor
        max_dim = os.getenv("PREPROCESS_SIZE")
        assert(max_dim is not None)
        preprocessor = model.ImagePreprocessor(int(max_dim))
    else:
        assert(False)

    print(f"Initialized {modelType}")

    server.add_insecure_port('[::]:50051')

    print("Starting server...")
    server.start() 
    print("Started!")
    
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
