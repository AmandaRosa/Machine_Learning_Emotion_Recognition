 # Código - Relação de emocoes com as coordenadas de estruturas do rosto
# Nome: Amanda Rosa Ferreira Jorge - 2022/1

# Classificador: SVM

#Legenda:
# 0 - Angry
# 1 - Disgusted
# 2 - Happy
# 3 - Neutral
# 4 - Sad
# 5 - Surprised
# 6 - Afraid
#----------------------------------------------------------------#

from sys import path_importer_cache
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, mean_squared_error, precision_score, recall_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from retinaface import RetinaFace
from scipy.spatial import distance
import cv2
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC
import pylab
import math
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn import svm
from itertools import cycle
import pickle

# ler o excel - 210 linhas de informacoes de 210 imagens com dados repetidos 3 vezes na tabela
df = pd.read_excel('Dados_CK_plus.xlsx')

Fotos = df.shape[0]
Bits = 9
#individuo = [0, 1, 1, 0, 0, 1, 0, 0, 1] #WSEFEP
individuo = [0, 1, 1, 1, 0, 0, 1, 1, 0] #CK_plus
#individuo = [1,1,1,1,1,1,1,1,1] #"sem ag"
#individuo = [1, 0, 0, 0, 1, 0, 0, 0, 1] #jaffe

vector_final = {}

# construir o dataframe
dataframe = pd.DataFrame(df.iloc[:,2:12])

print(dataframe)

# dividindo entre y - targets (rotulos) e X - dados independentes
X = dataframe.iloc[:,1:12]
y = dataframe.iloc[:,0]

vetor = np.array(X)

for x in range(Fotos):
    vector = []
    for i in range(Bits):
        vector_linha = individuo[i]* vetor[x][i]
        vector.append(vector_linha)

    vector_final[x] = [elem for elem in vector if elem != 0.0]

X_final = list(vector_final.values())

#Pre-processando dados
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size = 0.25,random_state=0)

## Feature Scaling
sc = StandardScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)

# Training the SVM Model on the training set
model = SVC(kernel= 'sigmoid', random_state=0) # kernel options can be: 'linear','poly', 'rbf', 'sigmoid', 'precomputed'
y_score = model.fit(X_train_norm, y_train).decision_function(X_test_norm)
y_pred = model.predict(X_test_norm)

filename = 'finalized_model_SVM.pkl'
pickle.dump(model, open(filename, 'wb'))
print('OK')