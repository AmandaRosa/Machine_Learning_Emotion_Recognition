 # Código - Relação de emocoes com as coordenadas de estruturas do rosto
# Nome: Amanda Rosa Ferreira Jorge - 2022/1

# Implementacao de Algoritmo Genético - Seleção de Parâmetros do Rosto

from Genetic_Algorithm import GA
from Dataframe import Dataframe
import matplotlib.pyplot as plt
import time
import os
import csv
import pickle

databases = ['JAFFE']
#databases = ['CK_plus', 'JAFFE']
methods = ['MLP', 'SVM']
Resultado_Final = {}

for database in databases:

    met = {}

    for method in methods:

        Individuos = 300 # 30
        Bits = 9
        Elite = 20 #15
        Qde_torneios = Individuos - Elite
        Grupo_torneio = 3
        Prob_crossover = 0.9
        Prob_mutacoes = 0.5
        Qde_crossover = int(Individuos/2)
        Qde_mutacoes = int(Individuos/2)
        Geracoes = 3 #30
        Metodo = method # MLP or SVM
        Best_fit = 0
        Best_ind = []
        Best_Accuracy_AG = []
        Best_Individuo_AG = []
        Lista = []
        fitness = []

        # Dados X and y
        X, y, Fotos = Dataframe.ler_dados(database)
        X = X.to_numpy()

        # inicializar minha população
        geracao_inicial = GA.init_pop(Individuos, Bits)
        #print(geracao_inicial)

        # Calcular fitness - Classificador
        fitness, Best_fit, Best_ind, Lista, model = GA.fitness_function(Nova_geracao, Individuos, Bits, X, y, Metodo, Fotos, Best_fit, Best_ind, Lista, classifier)

        # Coleta melhor individuo
        #Best_fit, Best_ind = GA.best(fitness, geracao_inicial, Best_fit, Best_ind

        # Realizar seleção X/2 melhores
        elite = GA.elitismo(fitness, geracao_inicial, Individuos, Elite)

        # Realizar torneio X/2
        torneio = GA.torneio(fitness, geracao_inicial, Individuos, Grupo_torneio, Qde_torneios)

        merged_pop = elite + torneio

        # População de Filhos - Crossover
        cross_pop = GA.crossover(merged_pop, Individuos, Qde_crossover, Prob_crossover, Bits)

        # Popualação de Filhos - Mutacao
        mut_pop = GA.mutation(cross_pop, Individuos, Qde_mutacoes, Prob_mutacoes, Bits)

        # Nova de Geração de Pais
        Nova_geracao = mut_pop

        print(Lista)
        met['Database'] = database
        met[method] = Lista[-1]
        Resultado_Final = met

        time.sleep(10)

        # Looping - N interações:
        for geracao in range(Geracoes-1):

            #print(Nova_geracao)
            #print(' ')

            # Calcular fitness - Classificador
            fitness, Best_fit, Best_ind, Lista, model = GA.fitness_function(Nova_geracao, Individuos, Bits, X, y, Metodo, Fotos, Best_fit, Best_ind, Lista, classifier)

            # Coleta melhor individuo
            #Best_fit, Best_ind = GA.best(fitness, geracao_inicial, Best_fit, Best_ind)

            # Realizar seleção X/2 melhores
            elite = GA.elitismo(fitness, Nova_geracao, Individuos, Elite)
            #print('ELITE',elite)

            # Realizar torneio X/2
            torneio = GA.torneio(fitness, Nova_geracao, Individuos, Grupo_torneio, Qde_torneios)

            merged_pop = elite + torneio
            #print('MERGED',merged_pop)

            # População de Filhos - Crossover
            cross_pop = GA.crossover(merged_pop, Individuos, Qde_crossover, Prob_crossover, Bits)
            #print('CROSS',cross_pop)

            # Popualação de Filhos - Mutacao
            mut_pop = GA.mutation(cross_pop, Individuos, Qde_mutacoes, Prob_mutacoes, Bits)

            # Nova de Geração de Pais
            Nova_geracao = mut_pop
            #print(Nova_geracao)

            time.sleep(10)

            print(Lista)
            
            met['Database'] = database
            met[method] = Lista[-1]
            Resultado_Final = met

        # salvar na tabela
        dados = {}
        aux = 1
        dados[aux] ={ 
                    'Individuos': str(Individuos),
                    'Elite': str(Elite),
                    'Geracoes': str(Geracoes),
                    'Resultado': Resultado_Final
        }
        
        #Registro.csv dos dados
        header = ['Individuos', 'Elite', 'Geracoes', 'Resultado']
        rows = [
        dados[aux]
        ]

        if os.path.isfile(f"Resultado.csv"):
            if os.stat(f"Resultado.csv").st_size == 0:
                with open(f"Resultado.csv", 'w', encoding='UTF8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                with open(f"Resultado.csv", 'a', encoding='UTF8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writerows(rows)
        else:
            with open(f"Resultado.csv", 'w', encoding='UTF8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)

    if database ==  'CK_plus':
    
        filename = f'finalized_model_{method}.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    else:
        pass

print(Resultado_Final)



