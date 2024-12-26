build: clean
	python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./src/proto/helloworld.proto
	python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./src/proto/ImageService.proto
clean:
	-rm ./src/proto/*.py
