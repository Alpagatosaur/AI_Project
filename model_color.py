# -*- coding: utf-8 -*-
# Create a model predicts color sprectrogram
import os
import glob
import sklearn
import numpy as np
import matplotlib.pyplot as plt

import librosa.display
import librosa
import splitfolders
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from methods import load_img, train_model

image_size = 100
MIN_IMGS_IN_CLASS=500

#Build array image spectrogram
n_mels=128

list_classes = ["bird", "cat", "cricket", "dog"] # = [0, 1, 2, etc.]
nb_classes = len(list_classes)

# PATH
path_file = os.getcwd()
os.path.dirname(os.path.abspath(path_file))

output_dir = os.path.join(path_file, 'train_test_color')
data_dir = os.path.join(path_file, 'Dataset')
img_dir = os.path.join(path_file, 'data_img_color')
model_dir = os.path.join(path_file, 'model_color.h5')

if os.path.exists(data_dir):
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
    
                # create save img spectrogram
                cmap = plt.get_cmap('hot')
                librosa.display.specshow(normalised_db, cmap=cmap)
                file_temp = path_output + "/" + list_classes[int_class] + cptNom + ".png"
                plt.savefig(file_temp, transparent=True)
                cpt += 1

# If test train folder dont exist
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
# model.add(VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size,3)))
model.add(Conv2D(32, 3, padding='same', activation='tanh', input_shape=(image_size, image_size,3)))
model.add(MaxPooling2D(padding='same'))
model.add(Conv2D(64, 3, padding='same', activation='tanh'))
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation='tanh'))
model.add(Dense(nb_classes, activation='softmax'))


# Train model
train_model(model, train_img, train_labels, "model_color")

# Test model
test_loss, test_acc = model.evaluate(test_img, test_labels)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
