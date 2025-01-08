import grpc
from concurrent import futures
import time

from S3Client import getFromS3, initS3
import model
from proto import ImageService_pb2
from pymilvus import MilvusClient
from proto import ImageService_pb2_grpc
from deepface import DeepFace

class ImageServicer(ImageService_pb2_grpc.ImageServiceServicer):
    def Identify(self, request, context):
        print(f"identify-{time.time()} {request}")

        response = ImageService_pb2.IdentifyResponse()

        classifier = model.ImageClassifier()

        print(request)
 
        encodedImage = getFromS3(request.base_image.url)
        print(request.base_image.url)

        embedding = classifier.extract_embedding(encodedImage, "Facenet512")

        for mfloat in embedding:
            response.embedding.append(mfloat)

        print("Response success!")
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    ImageService_pb2_grpc.add_ImageServiceServicer_to_server(ImageServicer(), server)
    initS3()
    # initialization trick
    DeepFace.build_model("Facenet512")
        
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
