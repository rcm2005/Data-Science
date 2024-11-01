#! /usr/bin/env python3 #
# -*- coding: utf-8 -*- #


## Importação dos módulos - - - - - - - - - - - - - - - - - - - - - - #
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics


## Script principal - - - - - - - - - - - - - - - - - - - - - - - - - #
## Objetivo: prever a idade de um abalone (molusco)
##           a idade pode ser obtido por contar o número de anéis

url = ('https://archive.ics.uci.edu/ml/machine-learning-databases'
       '/abalone/abalone.data')

## header=None indica que a primeira linha do arquivo csv
## não representa as colunas dos dados
data_frame = pd.read_csv(url, header=None)
## print(data_frame.head())

## atribuindo nomes mais significativos as colunas
data_frame.columns = [
    'sexo',
    'comprimento',
    'diâmetro',
    'altura',
    'peso total',
    'peso descascado',
    'peso das víceras',
    'peso da concha',
    'anéis'
]

## plotando um histograma com o número de anéis dos abalones
## queremos identificar se há algum padrão (concentração de dados)
data_frame['anéis'].hist(bins=15)
plt.show()
## do gráfico, sabemos que o número de anéis se concentra entre 5 a
## 15

## agora iremos verificar quais variáveis possuem um forte correlação
## com o número de anéis
## antes de computarmos a matriz de correlação, fazemos o
## pré-processamento da coluna sexo
label_encoder = LabelEncoder()
data_frame['sexo'] = label_encoder.fit_transform(data_frame['sexo'])

## computa e exibe a matriz de correlação
matriz_correlacao = data_frame.corr()
print(matriz_correlacao['anéis'])

figura, eixo = plt.subplots()
sns.heatmap(matriz_correlacao,
            annot=True)
plt.show()

## ao observarmos a matriz de correlação, certamente a coluna sexo
## deve ser removida
## a coluna peso descascado, é uma candidata a ser removida
data_frame.drop(columns=['sexo'], inplace=True)

## criação do modelo
## variáveis preditoras
X = data_frame.drop(columns=['anéis', 'peso descascado'])
## variável alvo
y = data_frame['anéis']

## fazendo a divisão dos dados em dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=12345)
## criação do modelo propriamente dito
modelo_knn = KNeighborsRegressor(n_neighbors=7)
modelo_knn.fit(X_train, y_train)

## avaliação do modelo
y_previsto = modelo_knn.predict(X_test)
print('\nAvaliação do modelo')
print('MAE:', metrics.mean_absolute_error(y_test, y_previsto))
print('MSE:', metrics.mean_squared_error(y_test, y_previsto))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_previsto)))

## comparação visual entre valor real (eixo y) x valor previsto
## (eixo x)
figura, eixo = plt.subplots()
eixo.scatter(y_previsto,
             y_test)
eixo.set_xlabel('Valor previsto')
eixo.set_ylabel('Valor real')
plt.show()
