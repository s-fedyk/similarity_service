import model
from pymilvus import MilvusClient
import os

# Set up a Milvus client
client = MilvusClient(db_name="default")
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

classifier = model.ImageClassifier()

print("inserting images...")
for imageName in os.listdir("images"):
    embedding = classifier.extract_embedding(f"images/{imageName}")
    client.insert(
        "image_embeddings",
        {"vector": embedding, "filename": imageName},
    )

print("all images inserted")
