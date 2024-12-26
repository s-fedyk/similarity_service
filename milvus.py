import model
from pymilvus import MilvusClient
import os

# Set up a Milvus client
client = MilvusClient(uri="example.db")
# Create a collection in quick setup mode
if client.has_collection(collection_name="image_embeddings"):
    client.drop_collection(collection_name="image_embeddings")

client.create_collection(
    collection_name="image_embeddings",
    vector_field_name="vector",
    dimension=4096,
    auto_id=True,
    enable_dynamic_field=True,
    metric_type="COSINE",
)

images = ["face1.jpg", "face2.jpg", "face3.jpg", "face4.jpg"]
classifier = model.ImageClassifier()

for imageName in os.listdir("images"):
    print(f"inserting {imageName}")
    embedding = classifier.extract_embedding(f"./images/{imageName}")
    if not embedding:
        continue
    client.insert(
        "image_embeddings",
        {"vector": embedding, "filename": imageName},
    )

print("all images inserted")
print("searching...")

embedding = classifier.extract_embedding(images[0])

query_vectors = [embedding]

res = client.search(
    collection_name="image_embeddings",  # target collection
    data=query_vectors,  # a list of one or more query vectors, supports batch
    limit=2,  # how many results to return (topK)
    output_fields=["vector", "filename"],  # what fields to return;
)

print(res)


