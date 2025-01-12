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

def resize_with_scaling(img):
# 1) Decode
        if img is None:
            raise ValueError("cv2.imdecode returned None. Possibly invalid image data.")

        orig_h, orig_w = img.shape[:2]

        # 2) Compute a scale factor so the image doesn’t exceed 1024×1024
        max_dim = 1024
        scale_w = 1.0
        scale_h = 1.0
        if orig_w > max_dim or orig_h > max_dim:
            # We’ll scale by whichever side is bigger
            # but you could also do a uniform approach
            if orig_w > orig_h:
                scale_w = max_dim / float(orig_w)
                scale_h = scale_w
            else:
                scale_h = max_dim / float(orig_h)
                scale_w = scale_h

        new_w = int(orig_w * scale_w)
        new_h = int(orig_h * scale_h)

        # 3) Resize if needed
        if new_w < orig_w or new_h < orig_h:
            print(f"Downscaling from {orig_w}x{orig_h} to {new_w}x{new_h}")
            img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # If we don’t need to downscale, just use the original
            img_scaled = img

        return img_scaled, scale_w, scale_h



class ImageClassifier(object):
    def __init__(self):
        DeepFace.build_model("Facenet512")
        DeepFace.build_model("retinaface", "face_detector")

        facenetClient= DeepFace.modeling.cached_models["facial_recognition"]["Facenet512"]
        facenetClient.model = download_and_extract_model("facenet512_neuron")

        #retinafaceClient = DeepFace.modeling.cached_models["face_detector"]["retinaface"]
        #retinafaceClient.model = download_and_extract_model("retinaface_neuron")

        return

    def extract_embedding(self, encodedImage, modelName="Facenet512"):
        print("Getting embedding...")
        result = None
        try: 
            nparr = np.frombuffer(encodedImage, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_scaled, scale_w, scale_h = resize_with_scaling(img)

            # 4) Pass the scaled image to DeepFace
            result = DeepFace.represent(
                img_scaled,
                enforce_detection=False,
                model_name=modelName,
                detector_backend="retinaface"
            )

        except Exception as e:
            print("Catching Exception!")
            print(e)

        if not result:
            print("No embedding!")
            return None

        print("Extracted!")
        
        first_face = result[0]
        facial_area = first_face["facial_area"]
        print(f"Scaled area : {facial_area}")

        scale_inv_x = 1.0 / scale_w
        scale_inv_y = 1.0 / scale_h

        facial_area["x"] = int(facial_area["x"] * scale_inv_x)
        facial_area["y"] = int(facial_area["y"] * scale_inv_y)
        facial_area["w"] = int(facial_area["w"] * scale_inv_x)
        facial_area["h"] = int(facial_area["h"] * scale_inv_y)

        left_eye_x = int(facial_area['left_eye'][0] * scale_inv_x) 
        left_eye_y = int(facial_area['left_eye'][1] * scale_inv_y) 

        right_eye_x = int(facial_area['right_eye'][0] * scale_inv_x) 
        right_eye_y = int(facial_area['right_eye'][1] * scale_inv_y) 

        facial_area['left_eye'] = (left_eye_x, left_eye_y)
        facial_area['right_eye'] = (right_eye_x, right_eye_y)

        print(f"Adjusted area : {facial_area}")
        print(f"Confidence : {first_face['face_confidence']}")

        embedding = first_face["embedding"]
        return embedding, facial_area
class FaceAnalyzer(object):
    def __init__(self):
        DeepFace.build_model("retinaface", "face_detector")
        DeepFace.build_model("Emotion", "facial_attribute")
        DeepFace.build_model("Age", "facial_attribute")
        DeepFace.build_model("Race", "facial_attribute")
        DeepFace.build_model("Gender", "facial_attribute")

        return

    def analyze_face(self, encodedImage):
        print("Analyzing...")
        analysis = None
        try: 
            nparr = np.frombuffer(encodedImage, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None. Possibly invalid image data.")

            orig_h, orig_w = img.shape[:2]

            # 2) Compute a scale factor so the image doesn’t exceed 1024×1024
            max_dim = 1024
            scale_w = 1.0
            scale_h = 1.0
            if orig_w > max_dim or orig_h > max_dim:
                # We’ll scale by whichever side is bigger
                # but you could also do a uniform approach
                if orig_w > orig_h:
                    scale_w = max_dim / float(orig_w)
                    scale_h = scale_w
                else:
                    scale_h = max_dim / float(orig_h)
                    scale_w = scale_h

            new_w = int(orig_w * scale_w)
            new_h = int(orig_h * scale_h)

            # 3) Resize if needed
            if new_w < orig_w or new_h < orig_h:
                print(f"Downscaling from {orig_w}x{orig_h} to {new_w}x{new_h}")
                img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                # If we don’t need to downscale, just use the original
                img_scaled = img

            analysis = DeepFace.analyze(img_scaled, enforce_detection=False, detector_backend="retinaface")

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
