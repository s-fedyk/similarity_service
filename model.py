import numpy as np
import cv2
from deepface import DeepFace
from deepface.modules.verification import find_distance, find_threshold

class ImageClassifier(object):
    def __init__(self):
        pass

    def preprocess(self, image):
        pre_processed_image = image
        return image

    def process(self, base_image, comparison_images):
        images = [base_image] + comparison_images
        pre_processed_images = list(map(self.preprocess, images))

        base_image = pre_processed_images[0]
        comparison_images = pre_processed_images[1:]

        results = []

        for comparison_image in comparison_images:
            result = DeepFace.verify(base_image, comparison_image)

            identified = False

            if result['verified']:
                identified = True
            
            results.append(identified)
        
        return results

    def extract_embedding(self, encodedImage, modelName="DeepFace"):
        print("Getting embedding...")
        try: 
            nparr = np.frombuffer(encodedImage, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            result = DeepFace.represent(img, enforce_detection=False, model_name=modelName)
        except Exception as e:
            print("Catching Exception!")
            print(e)


        if not result:
            print("No embedding!")
            return None

        print("Extracted!")
        return result[0]["embedding"]

if __name__ == "__main__":
    base_image = "face1.jpg"
    comparison_images = ["face2.jpg", "face3.jpg", "face4.jpg"]
    classifier = ImageClassifier()

    results = classifier.process(base_image, comparison_images)

    for result in results:
        print(result)

    embedding = classifier.extract_embedding("face1.jpg")
    
