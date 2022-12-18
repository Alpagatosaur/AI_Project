# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow import lite
from methods import load_img


path_file = os.getcwd()

model_dir = os.path.join(path_file, 'model.h5')
output_dir = os.path.join(path_file, 'train_test')


# Load test dataset
test_dir = os.path.join(output_dir, 'val')
list_test_img = glob.glob(test_dir + "/*/*.png")
test_img, test_labels = load_img(list_test_img)

# Load model
model = load_model(model_dir)

# Load the TFLite model and allocate tensors.
interpreter = lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test with 10 img
nb_test_img = 50

# RESULTS VAR
avg_acc_tflite = 0
avg_acc_model = 0


for i in range(nb_test_img):
    img_8 = test_img[i]
    label = test_labels[i]
    img_32 = img_8.astype('float32')
    img_32 = np.expand_dims(img_32, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_32)
    interpreter.invoke()
    output_tflite = interpreter.get_tensor(output_details[0]['index'])
    
    #output_model = model.predict(img_32)

    if output_tflite[0].argmax() == label:
        avg_acc_tflite += 1
    

    """if output_model[0].argmax() == label:
        avg_acc_model += 1"""
    
    print(" OUTPUT TFLITE ", output_tflite[0].argmax(), max(output_tflite[0]))
    #{ print(" OUTPUT MODEL  ", output_model[0].argmax(), max(output_model[0]))
    print(" LABEL  ", label)
    print("_______________________\n")


print(f'\nOUTPUT AVERAGE ACCURACY {100*avg_acc_tflite/nb_test_img:.2f} %')
# print(f'OUTPUT AVERAGE ACCURACY MODEL  {100*avg_acc_model/nb_test_img:.2f} %')
print("___________________________")

"""
OUTPUT AVERAGE ACCURACY TFLITE 78.00 %
OUTPUT AVERAGE ACCURACY MODEL  78.00 %
___________________________
"""