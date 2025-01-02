naive: build
	docker run -p 50051:50051 -v $(HOME)/.deepface/weights:/root/.deepface/weights pomidoro/similarity-service:1
dev:
	kubectl apply -f k8s/dev/configmap.yaml
	kubectl apply -f k8s/dev/deployment.yaml
	kubectl apply -f k8s/dev/service.yaml
prod:
	kubectl apply -f k8s/prod/deployment.yaml
	kubectl apply -f k8s/prod/service.yaml
teardown:
	- kubectl delete deployment similarity-service-deployment
	- kubectl delete service similarity-service
	- kubectl delete configmap similarity-service-config

build: clean
	python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/ImageService.proto
	docker build -t pomidoro/similarity-service:1 .
	docker push docker.io/pomidoro/similarity-service:1
clean:
	-rm ./proto/*.py
