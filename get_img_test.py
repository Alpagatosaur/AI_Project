# -*- coding: utf-8 -*-

# get an image for a specific class and number
import os
import glob
import sklearn
import numpy as np

import matplotlib.pyplot as plt
import librosa.display
import librosa
from PIL import Image

image_size = 100
MIN_IMGS_IN_CLASS=500

#Build array image spectrogram
n_mels=128

list_classes = ["bird", "cat", "cricket", "dog"] # = [0, 1, 2, etc.]
nb_classes = len(list_classes)

# PATH
path_file = os.getcwd()
os.path.dirname(os.path.abspath(path_file))

int_class = 2
data_dir = os.path.join(path_file, 'Dataset')
path_input = os.path.join(data_dir, list_classes[int_class])


glob_wav = glob.glob(path_input + "/*.wav")

cpt = 9
sound = glob_wav[cpt]


# bird\\XC109026.wav'

#Build array image
wave_data, wave_rate = librosa.load(sound)

if wave_data.shape[0]<5*wave_rate: # cut 5s
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
db_image.save(list_classes[int_class] + cptNom + ".png")

cmap = plt.get_cmap('hot')
librosa.display.specshow(mel, cmap=cmap)
plt.savefig(list_classes[int_class] + cptNom + "power_cmap.png")

cmap = plt.get_cmap('hot')
librosa.display.specshow(db_array, cmap=cmap)
plt.savefig(list_classes[int_class] + cptNom + "_cmap.png")
