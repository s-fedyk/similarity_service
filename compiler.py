from deepface import DeepFace
from deepface.models.facial_recognition.Facenet  import FaceNet512dClient, load_facenet512d_model 
from deepface.models.facial_recognition.ArcFace import *
import os
from deepface.modules.modeling import ArcFace
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
import numpy as np
from retinaface.model import retinaface_model


import tensorflow as tf
import tensorflow_neuron as tfn

# Load your .h5 model
os.environ["NEURON_CC_FLAGS"] = "--neuroncore-pipeline-cores=1 --extract-weights inf1.xlarge"
os.environ["NEURONCORE_GROUP_SIZES"] = "4"
os.environ["XLA_USE_BF16"] = "1"

# Build the RetinaFace model
model = tf.function(
    retinaface_model.build_model(),
    input_signature=(tf.TensorSpec(shape=[1, None, None, 3], dtype=np.float32),),
)


# Ensure proper backend settings
tf.keras.backend.set_learning_phase(0)
tf.keras.backend.set_image_data_format('channels_last')

# Define the example input tensor
example_input = tf.random.uniform((1, 1024, 1024, 3), dtype=tf.float32)

# Compile the wrapped model with Neuron
model_neuron = tfn.trace(model, example_inputs=example_input)

# Print the transformed operations ratio
print(f"Transformed {model_neuron.on_neuron_ratio} operations")

# Save the compiled Neuron model
model_neuron.save("retinaface_neuron")

print("Compiled RetinaFace!")

