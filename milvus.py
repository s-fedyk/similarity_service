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
    metric_type="L2",
)

classifier = model.ImageClassifier()

print("inserting images...")

files = os.listdir("img_align_celeba")
for idx,imageName in enumerate(files):
    print(f"{idx+1}/{len(files)}")
    with open(f"img_align_celeba/{imageName}", "rb") as file:
        contents = file.read()
        embedding = classifier.extract_embedding(contents, "VGG-Face")
        client.insert(
            "image_embeddings",
            {"vector": embedding, "filename": imageName, "model": "VGG-Face"},
        )

print("all images inserted")
