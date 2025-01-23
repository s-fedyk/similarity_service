from deepface.models.face_detection.RetinaFace import RetinaFaceClient
from retinaface import RetinaFace
import numpy as np
import retinaface
import S3Client
import gc
from PIL import Image, ExifTags
import tensorflow_neuron as tfn
import zipfile
import tempfile
from deepface.models.facial_recognition.Facenet  import FaceNet512dClient, load_facenet512d_model 
import tensorflow as tf
import cv2
import os
from deepface import DeepFace

def download_and_extract_model(model, extract_dir="./"):
    print(f"Downloading {model}")
    model_bytes = S3Client.getFromS3(f"{model}.zip", "similarity-model-store")
    if model_bytes is None:
        raise ValueError(f"Failed to download model: {model}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        tmp_file.write(model_bytes)
        tmp_file_path = tmp_file.name
        print(f"Model downloaded and saved to temporary file: {tmp_file_path}")
    
    with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print()
        print(f"Model extracted to directory: {extract_dir}")
    
    os.remove(tmp_file_path)
    print(f"Temporary file {tmp_file_path} removed.")

    model = tf.keras.models.load_model(f"{extract_dir}/{model}")

    print(model.summary())
    print("model download success!")

    return model


class ImagePreprocessor(object):
    def __init__(self, max_dim=800):
        DeepFace.build_model("retinaface", "face_detector")
        self.max_dim = max_dim
        retinafaceClient = DeepFace.modeling.cached_models["face_detector"]["retinaface"]
        retinafaceClient.model = download_and_extract_model("retinaface_neuron")

        self.retinafaceClient = retinafaceClient

    def resize_with_scaling(self, img):
        if img is None:
            raise ValueError("cv2.imdecode returned None. Possibly invalid image data.")

        orig_h, orig_w = img.shape[:2]
        max_dim = self.max_dim

        scale_w = max_dim / float(orig_w)
        scale_h = max_dim / float(orig_h)

        scale = min(scale_w, scale_h)

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        if scale < 1.0:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC  # or cv2.INTER_LINEAR

        img_scaled = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        pad_bottom = max_dim - new_h
        pad_right = max_dim - new_w

        img_scaled = cv2.copyMakeBorder(
            img_scaled,
            0, pad_bottom, 0, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Return scaled image and the scale factors
        return img_scaled, scale, scale, 0, 0


    def preprocess(self, encodedImage):
        print("decoding image...")
        nparr = np.frombuffer(encodedImage, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)        
        if img is None:
            raise ValueError("cv2.imdecode returned None. Possibly invalid image data.")

        img_scaled, scale_h, scale_w, _, _ = self.resize_with_scaling(img)

        print("try extraction")
        result = RetinaFace.detect_faces(img_scaled, 0.9, self.retinafaceClient.model, False)["face_1"]

        scale_inv_x = 1.0 / scale_h
        scale_inv_y = 1.0 / scale_w

        x_min = result["facial_area"][0]
        y_min = result["facial_area"][1]
        x_max = result["facial_area"][2]
        y_max = result["facial_area"][3]

        facial_area = result["landmarks"]

        print("Facial area is:")
        print(facial_area)

        facial_area["x"] = x_min * scale_inv_x
        facial_area["w"] = (x_max - x_min) * scale_inv_y
        facial_area["y"] = y_min * scale_inv_x
        facial_area["h"] = (y_max - y_min) * scale_inv_y

        print("converting eyes")
        left_eye = [facial_area["left_eye"][0] * scale_inv_x, facial_area["left_eye"][1] * scale_inv_y]
        right_eye = [facial_area["right_eye"][0] * scale_inv_x, facial_area["right_eye"][1] * scale_inv_y]

        facial_area["left_eye"] = left_eye
        facial_area["right_eye"] = right_eye

        print("eyes converted")

        face = img_scaled[y_min:y_max, x_min:x_max]
        print(f"shape size {len(face), len(face[0])}")

        print("finished extraction")

        return face, facial_area

class ImageClassifier(object):
    def __init__(self):
        DeepFace.build_model("Facenet512")
        DeepFace.build_model("retinaface", "face_detector")
        
        facenetClient= DeepFace.modeling.cached_models["facial_recognition"]["Facenet512"]
        facenetClient.model = download_and_extract_model("facenet512_neuron")

        retinafaceClient = DeepFace.modeling.cached_models["face_detector"]["retinaface"]
        retinafaceClient.model = download_and_extract_model("retinaface_neuron")

        self.retinafaceClient = retinafaceClient

        return

    def extract_embedding(self, encodedImage, modelName="Facenet512"):
        print("Getting embedding...")
        result = None
        try: 
            print("decoding image...")
            nparr = np.frombuffer(encodedImage, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            print(img.shape)
            result = DeepFace.represent(
                img, #TODO make this take in the face, 
                enforce_detection=False,
                model_name=modelName,
                detector_backend="skip"
            )

        except Exception as e:
            print("Catching Exception!")
            print(e)

        if not result:
            print("No embedding!")
            return None

        print("Extracted!")
        
        first_face = result[0]

        embedding = first_face["embedding"]
        
        return embedding
class FaceAnalyzer(object):
    def __init__(self):
        DeepFace.build_model("Emotion", "facial_attribute")
        DeepFace.build_model("Age", "facial_attribute")
        DeepFace.build_model("Race", "facial_attribute")
        DeepFace.build_model("Gender", "facial_attribute")

        return

    def analyze_face(self, encodedImage, actions=("emotion", "age", "gender", "race")):
        print("Analyzing...")
        analysis = None
        try: 
            nparr = np.frombuffer(encodedImage, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            analysis = DeepFace.analyze(img, enforce_detection=False,actions=actions, detector_backend="skip")

        except Exception as e:
            print("Catching Exception!")
            print(e)

        if not analysis:
            print("No Analysis!")
            return None

        print("Analyzed!")
        print(analysis)

        return analysis[0]

class FaceDetector(object):
    def __init__(self):
        DeepFace.build_model("retinaface", "face_detector")
        return

    def detect_face(self, encodedImage):
            print("Detecting face...")
            faces = None
            try: 
                nparr = np.frombuffer(encodedImage, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                faces = DeepFace.extract_faces(img, enforce_detection=False, detector_backend="retinaface")

            except Exception as e:
                print("Catching Exception!")
                print(e)

            if not faces:
                print("No Analysis!")
                return None

            print("Faces extracted!")
            print(faces)

            return faces[0]
