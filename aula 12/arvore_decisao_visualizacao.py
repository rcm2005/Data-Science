#! /usr/bin/env python3 #
# -*- coding: utf-8 -*- #


## Importação de módulos - - - - - - - - - - - - - - - - - - - - - - -#
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


## Parte principal do script - - - - - - - - - - - - - - - - - - - - -#
arquivo_csv = 'Social_Network_Ads.csv'
data_frame = pd.read_csv(arquivo_csv)

data_frame.drop(columns = ['User ID', 'Gender'], inplace=True)


## Treinamento do modelo - - - - - - - - - - - - - - - - - - - - - - -#
## variáveis preditoras
X = data_frame[['Age', 'EstimatedSalary']]
## variável alvo
Y = data_frame['Purchased']

## faz a divisão dos dados de treinamento e de teste
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42)
## fazendo as devidas transformações nos dados
escalonador = StandardScaler()
X_train = escalonador.fit_transform(X_train)
X_test = escalonador.transform(X_test)

## cria o modelo e faz o treinamento
modelo = tree.DecisionTreeClassifier(criterion='entropy',
                                     random_state = 0)
modelo.fit(X_train, Y_train)

## Fazendo algumas previsões - - - - - - - - - - - - - - - - - - - - -#
Y_previsto = modelo.predict(X_test)

## Avaliação do modelo - - - - - - - - - - - - - - - - - - - - - - - -#
print(f'Acurácia: {accuracy_score(Y_test, Y_previsto)}')
print(f'Relatório de classificação:\n'
      f'{classification_report(Y_test, Y_previsto)}')


## Visualizando a árvore de decisão - - - - - - - - - - - - - - - - - #
representacao_arvore = tree.export_text(modelo)

# figsize=(25,20)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(modelo,
                   feature_names=X.columns,
                   class_names=['No Purchased', 'Purchased'],
                   filled=True)
plt.show()
