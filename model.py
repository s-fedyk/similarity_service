from deepface.models.face_detection.RetinaFace import RetinaFaceClient
from retinaface import RetinaFace
import numpy as np
import retinaface
import S3Client
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

def resize_with_scaling(img):
    if img is None:
        raise ValueError("cv2.imdecode returned None. Possibly invalid image data.")

    orig_h, orig_w = img.shape[:2]

    max_dim = 1024
    scale_w = 1.0
    scale_h = 1.0
    if orig_w > max_dim or orig_h > max_dim:
        if orig_w > orig_h:
            scale_w = max_dim / float(orig_w)
            scale_h = scale_w
        else:
            scale_h = max_dim / float(orig_h)
            scale_w = scale_h

    new_w = int(orig_w * scale_w)
    new_h = int(orig_h * scale_h)

    if new_w < orig_w or new_h < orig_h:
        print(f"Downscaling from {orig_w}x{orig_h} to {new_w}x{new_h}")
        img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_scaled = img

    h, w = img_scaled.shape[:2]
    if h < max_dim or w < max_dim:
        pad_bottom = max_dim - h
        pad_right = max_dim - w

        img_scaled = cv2.copyMakeBorder(
            img_scaled,
            0, pad_bottom, 0, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  
        )
    else:
        img_scaled = img_scaled

    return img_scaled, scale_w, scale_h

def pil_to_cv2(image):
    # Convert a PIL image to an OpenCV format (BGR)
    image = np.array(image)
    # Convert RGB to BGR if needed
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def load_image_with_correct_orientation(encodedImage):
    from io import BytesIO
    image = Image.open(BytesIO(encodedImage))
    
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=="Orientation":
                break

        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print("Error handling EXIF orientation:", e)
    
    return pil_to_cv2(image)

class ImagePreprocessor(object):
    def __init__(self, max_dim=1024):
        self.max_dim = max_dim

    def preprocess(self, encodedImage):
        img = load_image_with_correct_orientation(encodedImage)
        if img is None:
            raise ValueError("cv2.imdecode returned None. Possibly invalid image data.")
        
        img_scaled, scale_w, scale_h = resize_with_scaling(img)

        return img_scaled, scale_w, scale_h

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
            print("try extraction")
            result = RetinaFace.detect_faces(img, 0.9, self.retinafaceClient.model, False)["face_1"]

            print(result)
            x_min = result["facial_area"][0]
            y_min = result["facial_area"][1]
            x_max = result["facial_area"][2]
            y_max = result["facial_area"][3]

            face = img[y_min:y_max, x_min:x_max]
            facial_area = result["landmarks"]

            face = img[y_min:y_max, x_min:x_max]            
            
            result = DeepFace.represent(
                face,
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
        print(f"Scaled area : {facial_area}")
        print(first_face)

        embedding = first_face["embedding"]

        facial_area["x"] = x_min
        facial_area["w"] = x_max - x_min
        facial_area["y"] = y_min
        facial_area["h"] = y_max - y_min

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

            analysis = DeepFace.analyze(img, enforce_detection=False, detector_backend="retinaface")

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
