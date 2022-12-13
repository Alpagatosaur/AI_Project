# -*- coding: utf-8 -*-
import os
from tensorflow.keras.models import load_model
from tensorflow import lite


path_file = os.getcwd()

model_dir = os.path.join(path_file, 'model.h5')

# Load model
model = load_model(model_dir)


# Convert the model
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# TIMER :  9 s

# Save the model TFLite
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
