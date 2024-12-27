import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

emotions = {
    'angry': 0,
    'happy': 1,
    'sad': 2,
    'surprise': 3,
    'neutral': 4
}


# Vetor de treinamento
X_train = []
Y_train = []

for dirpath, dirnames, filenames in os.walk("/root/.cache/kagglehub/datasets/samanyuk/fer2013/versions/1/archive (7)/train"):
    dirname = dirpath.split("/")[-1]

    for file in filenames:
        img= Image.open(os.path.join(dirpath, file))
        img_matriz= np.array(img)
        X_train.append(img_matriz)
        Y_train.append(emotions[dirname])


X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Vetor de teste
X_test = []
Y_test = []

for dirpath, dirnames, filenames in os.walk("/root/.cache/kagglehub/datasets/samanyuk/fer2013/versions/1/archive (7)/test"):
    dirname = dirpath.split("/")[-1]

    for file in filenames:
        img= Image.open(os.path.join(dirpath, file))
        img_matriz= np.array(img)

        X_test.append(img_matriz)
        Y_test.append(emotions[dirname])


X_test = np.array(X_test) 
Y_test = np.array(Y_test)


# Normalizando vetores
X_train= X_train / 255.0
X_test= X_test / 255.0

# Adicionando dimensão do canal
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)


print(X_train.shape, X_test.shape)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax') 
])


model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, Y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping], batch_size=64)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_acc}')



# Plotando gráficos de perda e acurácia
def plot_history(history):
    # perda
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss (train)')
    plt.plot(history.history['val_loss'], label='Loss (val)')
    plt.title('Loss durante o treinamento')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy (train)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (val)')
    plt.title('Acurácia durante o treinamento')
    plt.xlabel('Epochs')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)