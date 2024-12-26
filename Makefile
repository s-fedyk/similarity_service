naive: build
	docker run -p 50051:50051 -v $(HOME)/.deepface/weights:/root/.deepface/weights pomidoro/similarity-service:1
kube: build
	kubectl apply -f deployment.yaml
	kubectl apply -f service.yaml
teardown:
	- kubectl delete deployment similarity-service
	- kubectl delete service similarity-service
build: clean
	python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/ImageService.proto
	docker build -t pomidoro/similarity-service:1 .
	docker push docker.io/pomidoro/similarity-service:1
clean:
	-rm ./proto/*.py
