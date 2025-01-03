import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


train_data_path = "/root/.cache/kagglehub/datasets/samanyuk/fer2013/versions/1/archive (7)/train"
test_data_path = "/root/.cache/kagglehub/datasets/samanyuk/fer2013/versions/1/archive (7)/test"

emotions = {
    'happy': 0,
    'sad': 1,
    'angry': 2,
    'surprise': 3
}

# Vetor de treinamento
X_train = []
Y_train = []

for dirpath, dirnames, filenames in os.walk(train_data_path):
    dirname = dirpath.split("/")[-1]
    if dirname in emotions.keys():
        for file in filenames:
            img= Image.open(os.path.join(dirpath, file))
            img_matriz= np.array(img)
            X_train.append(img_matriz)
            Y_train.append(emotions[dirname])


X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train = np.expand_dims(X_train, axis=-1)

X_train = X_train / 255.0

print(X_train.shape, Y_train.shape)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)


print(X_train.shape, Y_train.shape)



# Vetor de teste
X_test = []
Y_test = []

for dirpath, dirnames, filenames in os.walk(test_data_path):
    dirname = dirpath.split("/")[-1]
    if dirname in emotions.keys():

        for file in filenames:
            img= Image.open(os.path.join(dirpath, file))
            img_matriz= np.array(img)

            X_test.append(img_matriz)
            Y_test.append(emotions[dirname])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_test = X_test / 255.0

print(X_train.shape, X_test.shape)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(len(emotions.keys()), activation='softmax')
])


model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(datagen.flow(X_train, Y_train, batch_size=32),
                    epochs=150,
                    validation_data=(X_test, Y_test),
                    )

test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_acc}')


# Plot perda
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss (train)')
plt.plot(history.history['val_loss'], label='Loss (val)')
plt.title('Loss durante o treinamento')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot acurácia
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy (train)')
plt.plot(history.history['val_accuracy'], label='Accuracy (val)')
plt.title('Acurácia durante o treinamento')
plt.xlabel('Epochs')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.show()

