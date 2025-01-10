from deepface.models.face_detection.RetinaFace import RetinaFaceClient
import numpy as np
import S3Client
import tensorflow_neuron as tfn
import zipfile
import tempfile
from deepface.models.facial_recognition.Facenet  import FaceNet512dClient, load_facenet512d_model 
import tensorflow as tf
import cv2
import os
from deepface import DeepFace

def download_and_extract_model(model, extract_dir="./"):
    """
    Downloads a model archive from S3 and extracts it to the specified directory.
    """
    print(f"Downloading {model}")
    model_bytes = S3Client.getFromS3(f"{model}.zip", "similarity-model-store")
    if model_bytes is None:
        raise ValueError(f"Failed to download model: {model}")
    
    # Save the model to a temporary zip file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        tmp_file.write(model_bytes)
        tmp_file_path = tmp_file.name
        print(f"Model downloaded and saved to temporary file: {tmp_file_path}")
    
    # Extract the zip file to the extract_dir
    with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print()
        print(f"Model extracted to directory: {extract_dir}")
    
    # Remove the temporary zip file
    os.remove(tmp_file_path)
    print(f"Temporary file {tmp_file_path} removed.")

    model = tf.keras.models.load_model(f"{extract_dir}/{model}")

    print(model.summary())
    print("model download success!")

    return model

class NeuronWrappedModel(tf.keras.Model):
    def __init__(self, neuron_model):
        super(NeuronWrappedModel, self).__init__()
        self.neuron_model = neuron_model

    def call(self, inputs):
        return self.neuron_model(inputs)

class ImageClassifier(object):
    def __init__(self):
        DeepFace.build_model("Facenet512")

        facenetClient= DeepFace.modeling.cached_models["facial_recognition"]["Facenet512"]
        
        facenetClient.model = download_and_extract_model("facenet512_neuron")
        print("model is setup")

        retinafaceClient = RetinaFaceClient()
        DeepFace.modeling.cached_models["face_detector"]["retinaface"] = retinafaceClient

        neuron_model = download_and_extract_model("retinaface_neuron")

        model = tf.function(
                    neuron_model,
                    input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),),
                )

        retinafaceClient.model = model

        return

    def extract_embedding(self, encodedImage, modelName="Facenet512"):
        print("Getting embedding...")
        result = None
        try: 
            nparr = np.frombuffer(encodedImage, np.uint8)
            print(len(nparr))
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(len(img))

            result = DeepFace.represent(img, enforce_detection=False, model_name=modelName, detector_backend="retinaface")
        except Exception as e:
            print("Catching Exception!")
            print(e)

        if not result:
            print("No embedding!")
            return None

        print("Extracted!")
        
        for res in result:
            print(f"Area : {res['facial_area']}")
            print(f"Confidence : {res['face_confidence']}")

        return result[0]["embedding"], result[0]["facial_area"]
