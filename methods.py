# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.optimizers import Adam
import cv2


image_size = 100
learning_rate = 0.0001
BATCH_SIZE = 64
EPOCHS = 10


# Resize img
def preprocess(my_img):
    height, width = my_img.shape[:2]
    scale = image_size / max(height, width)
    dx = (image_size - scale * width) / 2
    dy = (image_size - scale * height) / 2
    trans = np.array([[scale, 0, dx], [0, scale, dy]], dtype=np.float32)
    my_img = cv2.warpAffine(my_img, trans, (image_size, image_size), flags=cv2.INTER_AREA)
    my_img = cv2.resize(my_img, (image_size, image_size))
    return my_img

# Shuffle
def mixing(images, labels):
    images = np.array(images)
    labels = np.array(labels)
    s = np.arange(images.shape[0])
    np.random.seed(1337)
    np.random.shuffle(s)
    images=images[s]
    labels=labels[s]
    return images, labels


# Read all img in the folder
def load_img(list_path):
    list_images = []
    list_labels = []
    for img_path in list_path:
        path_split = img_path.split("\\")
        img = cv2.imread(img_path)
        list_images.append(preprocess(img))
        list_labels.append(int(path_split[-2]))
    return mixing(list_images, list_labels)

# Train model
def train_model(model, train_img, train_labels, name_model="model"):
    adam = Adam(lr=learning_rate)
    # SparseCategoricalCrossentropy == to provide labels as integers
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_img, train_labels,
                        batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, shuffle = True, verbose=1)

    # model.summary()
    model.save(name_model + '.h5')

    # Show evolution of accurracy/ loss depending on epoch
    fig, (ax0, ax1) = plt.subplots(2, sharex=True)
    ax0.plot(history.history['accuracy'], label = 'train_accuracy')
    ax0.plot(history.history['val_accuracy'], label = 'val_accuracy')
    ax0.set_xlabel('epoch')
    ax0.set_ylabel('accuracy')
    plt.legend()

    ax1.plot(history.history['loss'], label = 'train_loss')
    ax1.plot(history.history['val_loss'], label = 'val_loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    plt.legend()
    plt.show()