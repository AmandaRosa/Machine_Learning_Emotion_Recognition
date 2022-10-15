 # Código - Relação de emocoes com as coordenadas de estruturas do rosto
# Nome: Amanda Rosa Ferreira Jorge - 2022/1

# Classificador: Multi-Layer Perceptron

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
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from retinaface import RetinaFace
from scipy.spatial import distance
import cv2
from sklearn.preprocessing import minmax_scale
import math
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
import pickle

#from AG.Main import Individuos

# ler o excel - 210 linhas de informacoes de 210 imagens com dados repetidos 3 vezes na tabela
df = pd.read_excel('Dados_BIOLAB_v2.xlsx')

Fotos = df.shape[0]
Bits = 9
#individuo = [0, 0, 1, 1, 0, 1, 1, 1, 1] #WSEFE
#individuo = [0,0,1,1,0,1,1,1,1] #"melhor"
#individuo = [1,1,1,1,1,1,1,1,1] #"sem ag"
individuo = [0,0,0,1,0,0,0,1,1] #ck_plus
#individuo = [1, 1, 0, 1, 1, 0, 1, 0, 0] #jaffe
#individuo = [1,1,1,1,1,1,1,1,1] #"sem ag"

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

X_train, X_test, y_train, y_test = train_test_split(X_final, y, random_state=0, test_size=0.5)

## Feature Scaling
sc = StandardScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)

filename = 'finalized_model_MLP.pkl'

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(X_test_norm)
acc = loaded_model.score(X_test_norm, y_test)

print(acc)

acc_ = accuracy_score(y_test, y_pred)
print('Accuracy',acc_) 

mse = mean_squared_error(y_test, y_pred, squared=False)
print(mse)
rmse = math.sqrt(mse)
print(rmse)

print("f1_weighted:", metrics.f1_score(y_test, y_pred,average='weighted'))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)


cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Ficar em porcentagem
plt.subplots(figsize=(10,10))
sns.set(font_scale=1.5)
sns.heatmap(cmn, annot=True, fmt='.2f', vmin=0, vmax=1, cmap="YlGnBu_r")
plt.ylabel('Ground Truth', fontdict={'size':'20'})
plt.xlabel('Prediction', fontdict={'size':'20'})
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.rcParams.update({'font.size': 30})
plt.show()

