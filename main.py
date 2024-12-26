
import grpc
from concurrent import futures
import time

from proto import ImageService_pb2
from pymilvus import MilvusClient
from proto import ImageService_pb2_grpc

class ImageServicer(ImageService_pb2_grpc.ImageServiceServicer):
    def Identify(self, request, context):
        print(f"identify-{time.time()} {request}")
        base_image = request.base_image

        response = ImageService_pb2.IdentifyResponse()
        response.comparison_images.append(base_image)

        return response

def serve():
    # Create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add your servicer to the server
    ImageService_pb2_grpc.add_ImageServiceServicer_to_server(ImageServicer(), server)
    
    # Listen on port 50051
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    
    try:
        # Keep the server running
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
