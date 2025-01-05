from deepface.models.facial_recognition.VGGFace import VggFaceClient, os
import tensorflow as tf
import tensorflow_neuron as tfn

os.environ["NEURON_CC_FLAGS"] = "--verbose DEBUG"

# Load your .h5 model

model = VggFaceClient()

# Optional: check model summary
tf.keras.models.save_model(model.model, "base_model")

# Load the saved model
model = tf.keras.models.load_model("base_model")

input_shape = (1, 224, 224, 3)  # example shape
input_dtype = tf.float32

# Compile with Neuron
#  - dynamic_batch_size=True is optional, etc.
model_neuron = tfn.trace(model, example_inputs=[tf.random.uniform(input_shape, dtype=input_dtype)])
model_neuron.save("vgg_face_neuron")
print("Compiled!")
