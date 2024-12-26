import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import numpy as np
from PIL import Image


# Vetor de treinamento
X_train = []
Y_train = []

for dirpath, dirnames, filenames in os.walk("./data/train"):
    dirname = dirpath.split("/")[-1]

    for file in filenames:
        img= Image.open(os.path.join(dirpath, file))
        img_gray= img.convert('L')
        img_gray_matriz= np.array(img_gray)

        X_train.append(img_gray_matriz)
        Y_train.append(dirname)


X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Vetor de teste
X_test = []
Y_test = []

for dirpath, dirnames, filenames in os.walk("./data/test"):
    dirname = dirpath.split("/")[-1]

    for file in filenames:
        img= Image.open(os.path.join(dirpath, file))
        img_gray= img.convert('L')
        img_gray_matriz= np.array(img_gray)

        X_test.append(img_gray_matriz)
        Y_test.append(dirname)


X_test = np.array(X_test) 
Y_test = np.array(Y_test)


# Normalizando vetores
X_train= X_train.astype("float32") / 255.0
X_test= X_test.astype("float32") / 255.0

# Codificar rótulos
Y_test = LabelEncoder().fit_transform(Y_test)
Y_train = LabelEncoder().fit_transform(Y_train)



print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(X_test[4])
print(Y_test[4])


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Para classificação em 10 classes (dígitos de 0 a 9)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_acc}')



