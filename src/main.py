
import grpc
from concurrent import futures
import time

from proto import helloworld_pb2
from proto import helloworld_pb2_grpc

from proto import ImageService_pb2
from proto import ImageService_pb2_grpc


# Define a class to implement the server functions (derived from GreeterServicer)
class GreeterServicer(helloworld_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        # Implement the "SayHello" method
        name = request.name
        message = f"Hello, {name}!"
        return helloworld_pb2.HelloReply(message=message)

class ImageServicer(ImageService_pb2_grpc.ImageServiceServicer):
    def Identify(self, request, context):

        base_image = request.base_image
        comparison_images = request.comparison_images

        return ImageService_pb2.IdentifyResponse()


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
