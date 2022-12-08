# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
from tensorflow.keras.models import load_model
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

# Test with 10 img
nb_test_img = 10

for i in range(nb_test_img):
    img_8 = test_img[i]
    label = test_labels[i]
    img_32 = img_8.astype('float32')
    img_32 = np.expand_dims(img_32, axis=0)

    output_model = model.predict(img_32)

    print(" OUTPUT MODEL  ", output_model[0].argmax(), max(output_model[0]))
    print(" LABEL  ", label)
    print("_______________________\n")

"""
1/1 [==============================] - 0s 451ms/step
 OUTPUT MODEL   0 0.500328
 LABEL   0
_______________________

1/1 [==============================] - 0s 97ms/step
 OUTPUT MODEL   1 0.543564
 LABEL   1
_______________________

1/1 [==============================] - 0s 97ms/step
 OUTPUT MODEL   0 0.5383406
 LABEL   0
_______________________

1/1 [==============================] - 0s 96ms/step
 OUTPUT MODEL   0 0.6007461
 LABEL   0
_______________________

1/1 [==============================] - 0s 88ms/step
 OUTPUT MODEL   2 0.73834866
 LABEL   2
_______________________

1/1 [==============================] - 0s 94ms/step
 OUTPUT MODEL   1 0.5029472
 LABEL   0
_______________________

1/1 [==============================] - 0s 102ms/step
 OUTPUT MODEL   0 0.54259557
 LABEL   0
_______________________

1/1 [==============================] - 0s 103ms/step
 OUTPUT MODEL   1 0.5450029
 LABEL   1
_______________________

1/1 [==============================] - 0s 95ms/step
 OUTPUT MODEL   0 0.50884855
 LABEL   0
_______________________

1/1 [==============================] - 0s 110ms/step
 OUTPUT MODEL   2 0.7887275
 LABEL   2
_______________________
"""