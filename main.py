from deepface.models.facial_recognition.Facenet import os
import grpc
from concurrent import futures
import time

from S3Client import getFromS3, initS3
import S3Client
import model
from proto import ImageService_pb2
from proto import ImageService_pb2_grpc
from proto import Analyzer_pb2
from proto import Analyzer_pb2_grpc

# singleton
embedder = None
analyzer = None

class ImageServicer(ImageService_pb2_grpc.ImageServiceServicer):
    def Identify(self, request, context):
        print(f"identify-{time.time()} {request}")
        global embedder

        encodedImage = getFromS3(request.base_image.url, S3Client.bucket_name)

        embedding,faceArea = embedder.extract_embedding(encodedImage, "Facenet512")

        response = ImageService_pb2.IdentifyResponse()
        response.embedding.extend(embedding)

        response.facial_area.w = faceArea["w"]
        response.facial_area.h = faceArea["h"]
        response.facial_area.x = faceArea["x"]
        response.facial_area.y = faceArea["y"]

        response.facial_area.left_eye.x = faceArea["left_eye"][0]
        response.facial_area.left_eye.y = faceArea["left_eye"][1]

        response.facial_area.right_eye.x = faceArea["right_eye"][0]
        response.facial_area.right_eye.y = faceArea["right_eye"][1]

        print(f"Responding with: {response}")
        return response

class AnalysisServicer(Analyzer_pb2_grpc.AnalyzerServicer):
    def Analyze(self, request, context):
        print(f"identify-{time.time()} {request}")
        global analyzer

        encodedImage = getFromS3(request.base_image.url, S3Client.bucket_name)

        analysis = analyzer.analyze_face(encodedImage)

        response = Analyzer_pb2.AnalyzeResponse()

        response.age = str(analysis["age"])
        response.gender = analysis["dominant_gender"]
        response.race = analysis["dominant_race"]
        response.emotion = analysis["dominant_emotion"]

        print(f"Responding with: {response}")
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    modelType = os.getenv("MODEL_TYPE")
    validTypes = ["embedder", "analyzer"]
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
    else:
        assert(False)

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
