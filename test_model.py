import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt



modelo = load_model('model.h5')

def proc_img(imagem_path):
    # Carregar imagem
    img= image.load_img(imagem_path, target_size=(48, 48), color_mode='grayscale')

    img_array = image.img_to_array(img) / 255.0

    # Adicionar a dimensão do batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def previsao(imagem_path):
    img_array = proc_img(imagem_path)

    # vetor de probabilidades
    predicao = modelo.predict(img_array)

    return predicao


emotions= ['happy', 'sad', 'neutral']

files = os.listdir("./test/")

for filename in files:
    path = os.path.join('./test/', filename)
 
    pred = previsao(path)

    print()
    print(f"Analizando a imagem: {filename}")
    plt.imshow(image.load_img(path))
    plt.axis('off')  
    plt.show()
    print()
    print(f'A classe prevista é: \033[1m{emotions[np.argmax(pred)]}\033[0m')
    print(f'Vetor de probabilidades: ')
    
    for i in range(3):
        print(f'Probabilidade de ser {emotions[i]}: {pred[0][i]*100:.2f}%')

    print()