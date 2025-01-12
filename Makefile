naive: build
	docker run -p 50051:50051 -v $(HOME)/.deepface/weights:/root/.deepface/weights pomidoro/similarity-service:1
dev:
	kubectl apply -f k8s/dev/embedder/configmap.yaml
	kubectl apply -f k8s/dev/embedder/deployment.yaml
	kubectl apply -f k8s/dev/embedder/service.yaml
	kubectl apply -f k8s/dev/analyzer/configmap.yaml
	kubectl apply -f k8s/dev/analyzer/deployment.yaml
	kubectl apply -f k8s/dev/analyzer/service.yaml
prod: aprod eprod
aprod:
	kubectl apply -f k8s/prod/analyzer/configmap.yaml
	kubectl apply -f k8s/prod/analyzer/deployment.yaml
	kubectl apply -f k8s/prod/analyzer/service.yaml
eprod:
	kubectl apply -f k8s/prod/embedder/configmap.yaml
	kubectl apply -f k8s/prod/embedder/deployment.yaml
	kubectl apply -f k8s/prod/embedder/service.yaml
tearanalyzer:
	- kubectl delete deployment analyzer-service-deployment
	- kubectl delete service analyzer-service
	- kubectl delete configmap analyzer-service-config
tearembedder:
	- kubectl delete deployment similarity-service-deployment
	- kubectl delete service similarity-service
	- kubectl delete configmap similarity-service-config
teardown: tearanalyzer tearembedder
build: clean
	docker build -t pomidoro/similarity-service:1 .
	docker push docker.io/pomidoro/similarity-service:1
clean:
	-rm ./proto/*.py
proto: clean
	python3.10 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/ImageService.proto
	python3.10 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/Analyzer.proto
	python3.10 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/Preprocessor.proto
