# Código - Relação de emocoes com as coordenadas de estruturas do rosto
# Nome: Amanda Rosa Ferreira Jorge - 2022/1

#Legenda:
# 0 - Angry
# 1 - Disgusted
# 2 - Happy
# 3 - Neutral
# 4 - Sad
# 5 - Surprised
# 6 - Afraid
#----------------------------------------------------------------#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from retinaface import RetinaFace
from scipy.spatial import distance
import cv2
import time
import os
import csv

# Neste codigo vamos extrair informações das imagens e completar a tabela com informações
# média, variância, distancias euclidianas entre as coordenadas do rosto

#Tabela
# Imagem - Emocao - Categoria - Media - Variancia - OD_BD - OE_BE - N_BE - N_BD - N_OE - N_OD

emocoes = ['Angry', 'Disgusted', 'Happy', 'Neutral', 'Sad', 'Surprised', 'Afraid']

### Coordenadas das faces das fotos -------------------

for emocao in emocoes:
    index = 1
    name = f'./Faces_images/{emocao}/picture ({index}).jpg'
    im = cv2.imread(name)

    while im is not None:

        # ler imagens da pasta
        name = f'./Faces_images/{emocao}/picture ({index}).jpg'
        im = cv2.imread(name)

        if im is None: #caso a leitura da imagem nao existir
            break

        categoria = emocoes.index(emocao)

        #detecta face 
        face = RetinaFace.detect_faces(name)

        facial_area = face['face_1']["facial_area"]
        y=facial_area[0]
        x=facial_area[1]
        h=facial_area[2]
        w=facial_area[3]

        #calculo da media da face
        media_face = np.mean(im[x:w, y:h])

        #calculo da variancia da face
        var_face = np.var(im[x:w, y:h])

        #calculo das distancias euclidianadas da face
        OD = (face['face_1']["landmarks"]["right_eye"])
        OE = (face['face_1']["landmarks"]["left_eye"])
        N = (face['face_1']["landmarks"]["nose"])
        BD = (face['face_1']["landmarks"]["mouth_right"])
        BE = (face['face_1']["landmarks"]["mouth_left"])

        #OD_BD
        OD_BD = distance.euclidean(OD, BD)

        #OE_BE
        OE_BE = distance.euclidean(OE, BE)

        #N_BE
        N_BE = distance.euclidean(N, BE)

        #N_BD
        N_BD = distance.euclidean(N, BD)

        #N_OE
        N_OE = distance.euclidean(N, OE)

        #N_OD
        N_OD = distance.euclidean(N, OD)

        #BE_BD
        BE_BD = distance.euclidean(BE, BD)

        dados = {
            'Emocao': str(emocao),
            'Path': str(name),
            'Categoria': categoria,
            'Media': media_face,
            'Variancia': var_face,
            'OD_BD': OD_BD,
            'OE_BE': OE_BE,
            'N_BE': N_BE,
            'N_BD': N_BD,
            'N_OE': N_OE,
            'N_OD': N_OD,
            'BE_BD': BE_BD
        }

        #Registro.csv e imagens local
        header = ['Emocao', 'Path', 'Categoria','Media', 'Variancia',  
                    'OD_BD', 'OE_BE', 'N_BE', 'N_BD', 'N_OE', 'N_OD', 'BE_BD']
        rows = [
            dados
        ]

        if os.path.isfile(f"Dados.csv"):
            if os.stat(f"Dados.csv").st_size == 0:
                with open(f'Dados.csv', 'w', encoding='UTF8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                with open(f'Dados.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writerows(rows)
        else:
            with open(f'Dados.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)   
        index +=1
        print('OK')

    print('OK')
print('OK')

