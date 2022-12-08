# -*- coding: utf-8 -*-
import os
import glob
import sklearn
import numpy as np

import librosa
import splitfolders
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from methods import load_img, train_model

image_size = 100
MIN_IMGS_IN_CLASS=500

#Build array image
n_fft=2040
hop_length=512
n_mels=128
fmin=20
fmax=8000
top_db=80

list_classes = ["cat","dog", "owl"] # = [0, 1, 2]
nb_classes = len(list_classes)

# PATH
path_file = os.getcwd()
os.path.dirname(os.path.abspath(path_file))

output_dir = os.path.join(path_file, 'train_test')
data_dir = os.path.join(path_file, 'Dataset')
img_dir = os.path.join(path_file, 'data_img')
model_dir = os.path.join(path_file, 'model.h5')

if not os.path.exists(img_dir):
    os.mkdir(img_dir)
    for int_class in range(len(list_classes)) :
        path_input = os.path.join(data_dir, list_classes[int_class])
        path_output = os.path.join(img_dir, str(int_class))
        os.mkdir(path_output)
        
        glob_wav = glob.glob(path_input + "/*.wav")
        cpt = 0
        for sound in glob_wav:
            #Build array image
            wave_data, wave_rate = librosa.load(sound)
            
            if wave_data.shape[0]<5*wave_rate:
              wave_data=np.pad(wave_data,int(np.ceil((5*wave_rate-wave_data.shape[0])/2)),mode='reflect')
            else:
              wave_data=wave_data[:5*wave_rate]
            
            
            #The variable below is chosen mainly to create a 216x216 image
            cptNom = str(cpt)
            mel = librosa.feature.melspectrogram(wave_data, sr=wave_rate, n_mels=n_mels)
            db = librosa.power_to_db(mel)
            normalised_db = sklearn.preprocessing.minmax_scale(db)
            db_array = (np.asarray(normalised_db)*255).astype(np.uint8)
            db_image =  Image.fromarray(np.array([db_array, db_array, db_array]).T)
            db_image.save(path_output + "/" + list_classes[int_class] + cptNom + ".png")
            cpt += 1


if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    splitfolders.ratio(img_dir, output=output_dir, seed=1337, ratio=(.8, .2))
    

# Load test dataset
test_dir = os.path.join(output_dir, 'val')
list_train_img = glob.glob(test_dir + "/*/*.png")
test_img, test_labels = load_img(list_train_img)


# Load train dataset
train_dir = os.path.join(output_dir, 'train')
list_train_img = glob.glob(train_dir + "/*/*.png")
train_img, train_labels = load_img(list_train_img)


# Create the model
model = Sequential()
model.add(VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size,3)))
model.add(Conv2D(16, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(padding='same'))
model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(padding='same'))
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(padding='same'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))


# Train model
train_model(model, train_img, train_labels)

# Test model
test_loss, test_acc = model.evaluate(test_img, test_labels)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)