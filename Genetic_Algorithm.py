# Código - Relação de emocoes com as coordenadas de estruturas do rosto
# Nome: Amanda Rosa Ferreira Jorge - 2022/1

# Implementacao de Algoritmo Genético - Seleção de Parâmetros do Rosto


from heapq import merge
from pickle import TRUE
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class GA:
    
    def init_pop(Populacao, Bits):

        Geracao = []
        for i in range(Populacao):
            individuo=list((np.random.random(Bits) < 1/3.).astype(int))
            matches = [x for x in individuo if x==1]
            if not matches:
                posicao = random.sample(range(0, Bits), 1)
                posicao = posicao[0]
                individuo[posicao] = 1
            Geracao.append(individuo)
        
        return Geracao

    def fitness_function(Geracao_inicial, Individuos, Bits, X, y, Metodo, Fotos, Best_fit, Best_ind, Lista): 

        fitness = []
        vector_final = {}
        vetor = np.array(X)
        #print(Geracao_inicial)

        for individuo in Geracao_inicial:
            #print(individuo)
            for x in range(Fotos):
                vector = []
                for i in range(Bits):
                    vector_linha = individuo[i]* vetor[x][i]
                    vector.append(vector_linha)

                vector_final[x] = [elem for elem in vector if elem != 0.0]

            X_final = list(vector_final.values())
            
            X_train, X_test, y_train, y_test = train_test_split(X_final, y, random_state=0, test_size=0.25)

            ## Feature Scaling
            sc = StandardScaler()
            X_train_norm = sc.fit_transform(X_train)
            X_test_norm = sc.transform(X_test)
            
            if Metodo == 'MLP':

                # Multi-Layer Perceptron de 3 camadas com 30 neuronios
                classifier = MLPClassifier(random_state=0, max_iter=4000, hidden_layer_sizes = (30,30,30,13,))
                #classifier = MLPClassifier(random_state=0, max_iter=3500, hidden_layer_sizes = (40,40,40,))
                classifier.fit(X_train_norm, y_train)
                y_pred = classifier.predict(X_test_norm)

                # Acurácia
                acc = accuracy_score(y_test, y_pred)
                #print('Accuracy',acc, 'Individuo', individuo)
                #print('Best_fit', Best_fit, 'Best_ind', Best_ind)

                #print(acc, Best_fit)
                if acc>Best_fit:
                    new = []
                    #print('CHANGED!')
                    Best_fit = acc
                    Best_ind = individuo
                    new.append(round(Best_fit,4))
                    new.append(str(Best_ind))
                    Lista.append(new)
                    print(Best_fit, Best_ind) 
                else:
                    pass
       
            elif Metodo == 'SVM':
                # Training the SVM Model on the training set
                classifier = SVC(kernel= 'sigmoid', random_state=0) # kernel options can be: 'linear','poly', 'rbf', 'sigmoid', 'precomputed'
                y_score = classifier.fit(X_train_norm, y_train).decision_function(X_test_norm)
                y_pred = classifier.predict(X_test_norm)

                # Acurácia
                acc = accuracy_score(y_test, y_pred)
                #print('Accuracy',acc, 'Individuo', individuo)

                if acc>Best_fit:
                    new = []
                    #print('CHANGED!')
                    Best_fit = acc
                    Best_ind = individuo
                    new.append(round(Best_fit,4))
                    new.append(str(Best_ind))
                    Lista.append(new)
                    print(Best_fit, Best_ind) 
                else:
                    pass
            
            fitness.append(acc)
        return fitness, Best_fit, Best_ind, Lista, classifier

       
    def elitismo(fitness, geracao_inicial, Individuos, Elite):
        # selecao dos melhores individuos da minha população. Se eu considerar uma população de X indivíduos vou selecionar X/2
        fitness = np.array(fitness)

        Pais = []
        Posicao = []
        x = 0
        # ordena ordem decrescente o vetor fitness
        auxiliar = sorted(fitness, reverse=True)
        auxiliar = np.array(auxiliar)
        #print(auxiliar)

        
        for i in auxiliar:
            if x == i:
                pass
            else:
                x = i
                index = np.where(fitness == i)
                index = np.asarray(index)
                lista = index[0]
                for element in lista:
                    Posicao.append(element)

        for i in Posicao:
            Pais.append(geracao_inicial[i])
        
        Pais = Pais[0:Elite]    
        
        return Pais
        
    def torneio(fitness, geracao_inicial, Individuos, Grupo_torneio, Qde_torneios):
        Torneio = []

        for i in range(Qde_torneios):
            
            #selecionados randomicamente 3 individuos para o torneio
            Escolhidos = random.sample(range(0, Individuos), Grupo_torneio)
            Resultado = []

            for x in Escolhidos:
                aux = fitness[x]
                Resultado.append(aux)
            
            index = Resultado.index(max(Resultado))
            vencedor  = Escolhidos[index]
            Torneio.append(geracao_inicial[vencedor])   
        
        return Torneio          

    def crossover(merged_pop, Individuos, Qde_crossover, Prob_crossover, Bits):
        
        cross_pop = merged_pop
        # dentro dos filhos selecionados, realizar o crossover com um taxa de C

        for i in range(Qde_crossover):
            prob = random.random()

            if prob <= Prob_crossover:

                # selecao de posicao aleatoria no cromossomo
                posicao = random.sample(range(0, Bits), 1)
                posicao = posicao[0]
                #print(posicao)

                # selecao aleatoria dos pais para crossover 
                ind_1 = random.sample(range(0, Individuos), 1)
                ind_2 = random.sample(range(0, Individuos), 1)

                pai_1 = merged_pop[ind_1[0]]
                pai_2 = merged_pop[ind_2[0]]
                #print(pai_1)
                #print(pai_2)

                # crossover
                aux_1 = pai_1[posicao:len(pai_1)]
                aux_2 = pai_2[posicao:len(pai_2)]
                pai_1[posicao:len(pai_1)] = aux_2
                pai_2[posicao:len(pai_2)] = aux_1
                #print(pai_1)
                #print(pai_2)

                cross_pop[ind_1[0]] = pai_1
                cross_pop[ind_2[0]] = pai_2


        return cross_pop
        

    def mutation(cross_pop, Individuos, Qde_mutacoes, Prob_mutacoes, Bits):
         # dentro dos filhos selecionados, realizar o crossover com um taxa de C
        mut_pop = cross_pop

        for i in range(Qde_mutacoes):
            prob = random.random()

            if prob <= Prob_mutacoes:
                '''
                ind_1 = random.sample(range(0, Individuos), 1)
                cross_pop[ind_1[0]] = list((np.random.random(Bits) < 1/3.).astype(int))
                '''
                # selecao de posicao aleatoria no cromossomo
                posicao = random.sample(range(0, Bits), 1)
                posicao = posicao[0]
                #print(posicao)

                # selecao aleatoria dos pais para crossover 
                ind_1 = random.sample(range(0, Individuos), 1)

                pai_1 = cross_pop[ind_1[0]]
                #print(pai_1)
                bit = pai_1[posicao]

                # mutacao
                pai_1[posicao] = int(not(bit))
                #print(pai_1)

                mut_pop[ind_1[0]] = pai_1

        return mut_pop

    def best(fitness, geracao, Best_fit, Best_ind):
        for i in fitness:
            if i > Best_fit:
                Best_fit = i
                index = np.where(fitness == i)
                index = np.asarray(index)
                index = index[0][0]
                Best_ind = geracao[index]
        return (Best_fit, Best_ind)




    