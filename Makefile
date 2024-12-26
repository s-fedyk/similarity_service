build: clean
	python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/ImageService.proto
	docker build -t my-grpc-milvus-app .
	docker run -p 50051:50051 -v $(HOME)/.deepface/weights:/root/.deepface/weights my-grpc-milvus-app
clean:
	-rm ./proto/*.py
