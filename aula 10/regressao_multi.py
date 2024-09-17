#! /usr/bin/env python3 #
# -*- coding: utf-8 -*- #


## Importação de módulos - - - - - - - - - - - - - - - - - - - - - - -#
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


## Script principal - - - - - - - - - - - - - - - - - - - - - - - - - #
## Etapa 0 - Carregar os dados - - - - - - - - - - - - - - - - - - - -#
## carrega os dados a partir de um csv
arquivo_csv = "insurance.csv"
data_frame = pd.read_csv(arquivo_csv)

## Etapa 1 - Examinar os dados - - - - - - - - - - - - - - - - - - - -#
## faz a primeir visualização dos dados
print(data_frame.head(), end="\n\n")
print(data_frame.describe())

## é possível verificar que existem variáveis categóricas
## antes de aplicarmos a regressão múltipla, vamos tratar as
## variáveis categóricas

## Etapa 2 - Pré-processamento - - - - - - - - - - - - - - - - - - - -#
## trocas as classes por uma enumeração de 0 até o número de classes -1
le = LabelEncoder()

## antes de aplicar o encoder
# print(data_frame['sex'].head())

## o método fit determinar uma enumeração das classes
## o método transform aplica a enumeração num campo
le.fit(data_frame.sex)
data_frame.sex = le.transform(data_frame.sex)

## depois de aplicar o encoder
## veja que 0 -> female, 1 -> male
# print(data_frame['sex'].head())

le.fit(data_frame.smoker)
data_frame.smoker = le.transform(data_frame.smoker)

## a instrução abaixo mostra que há 4 regiões distintas
# print(data_frame['region'].value_counts())
le.fit(data_frame.region)
data_frame.region = le.transform(data_frame.region)

## exibe um resumo dos dados após as transformações
print(data_frame.info())

## Etapa 3 - Econtrar relações entre as variáveis - - - - - - - - - - #
## obtém uma tabela de correlação das variáveis
tabela_correlacao = data_frame.corr()
print(tabela_correlacao)

## visualização gráfica da tabela de correlação
figura, eixo = plt.subplots(figsize=(10,8))
sns.heatmap(tabela_correlacao,
            cmap=sns.color_palette("Blues"),
            linewidths=.5,
            annot=True)
plt.show()

## analisando a distribuição dos gastos
plt.figure(figsize=(12,5))
plt.title("Distribuição dos gastos")
eixo = sns.histplot(data_frame['charges'], color='b')
plt.show()

## analisando os gastos para fumantes e não fumantes
fig_fumantes = plt.figure(figsize=(17,6))

## o argumento 121 indica que há
## 1 linha, 2 colunas e a figura tem o índice
## (ocupa a primeira posião da esquerda para direita)
## cria o gráfico para fumantes
eixo = fig_fumantes.add_subplot(121)
sns.histplot(data_frame[ (data_frame.smoker == 1) ]['charges'],
             color='r',
             ax=eixo,)
eixo.set_title('Distribuição dos gastos dos fumantes')

## cria o gráfico para não fumantes
eixo = fig_fumantes.add_subplot(122)
sns.histplot(data_frame[ (data_frame.smoker == 0) ]['charges'],
             color='b',
             ax=eixo)
eixo.set_title('Distribuição dos gastos dos não fumantes')

plt.show()

## analisando a distribuição de fumantes e não fumantes por sexo
fig_fumante_sexo = sns.catplot(x='smoker',
                               kind='count',
                               hue='sex',
                               palette="Blues_r",
                               data=data_frame,
                               legend_out=True)

#fig_fumante_sexo.set_axis_labels("", "Total").set_xticklabels(["Não Fuamnte"], ["Fumante"])
fig_fumante_sexo._legend.set_title('Sexo')
new_labels = ['Mulheres', 'Homens']

for t, l in zip (fig_fumante_sexo._legend.texts, new_labels):
    t.set_text(l)

plt.show()

## analisando os dados dos contratanes por idade
plt.figure(figsize=(12,5))
plt.title("Distribuição por idade")
eixo = sns.histplot(data_frame["age"], color='b')
plt.show()

## analisando a distribuição dos custos dos fumantes por idade
plt.figure(figsize=(12,5))
plt.title("Distribuição de custos por idade e por fumantes")
sns.scatterplot(x=data_frame.age,
                y=data_frame.charges,
                hue=data_frame.smoker,
                sizes=(12,5),
                palette="ch:r=-.2,d=.3_r")
plt.show()

## analisnado o IMC, ou BMI, nos custos hospitalares
plt.figure(figsize=(12,5))
plt.title("Distribuição de IMC")
eixo = sns.histplot(data_frame['bmi'], color='b')
plt.show()

plt.figure(figsize=(12,5))
plt.title("Distribuição de custos com pacientes com IMC maior que 30")
eixo = sns.histplot(data_frame[ (data_frame.bmi >= 30) ]['charges'],
                    color='b')
plt.show()

plt.figure(figsize=(12,5))
plt.title("Distribuição de custos com pacientes com IMC menor que 30")
eixo = sns.histplot(data_frame[ (data_frame.bmi < 30) ]['charges'],
                    color='b')
plt.show()

## analisando o número de filhos x custo hospitalar
sns.catplot(x="children",
            kind="count",
            palette="Blues",
            data=data_frame)
plt.show()

plt.figure(figsize=(12,5))
plt.title("Distribuição de custos por número de filhos")
sns.scatterplot(x=data_frame.children,
                y=data_frame.charges,
                sizes=(12,5))
plt.show()

## Etapa 4 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
## Para construi o modelo, vamos separar as variáveis preditoras da   #
## variável alvo                                                      #
df_var_preditoras = data_frame.drop(['charges'], axis=1)
df_var_alo = data_frame.charges

## criando os dados de treino e teste
## test_size=0.2 indica 80% dos dados para treino e 20% para teste
x_train, x_test, y_train, y_test = train_test_split(df_var_preditoras,
                                                   df_var_alo,
                                                   test_size=0.2,
                                                   random_state=0)
modelo_reg_multi = LinearRegression()
## treina o modelo (obtém os coeficientes) a partir dos dados de treino
modelo_reg_multi.fit(x_train, y_train)

## este coeficiente diz o quanto um modelo é capaz de se ajustar a amostra
## ele varia de 0 a 1, quanto mais próximo de 1 mais o modelo se ajusta a
## amostra
r_sq = modelo_reg_multi.score(df_var_preditoras, df_var_alo)
print(f"Coeficiente de Determinação (R^2): {r_sq}")

print(f"Intercepto: {modelo_reg_multi.intercept_}")
coeficientes = pd.DataFrame(modelo_reg_multi.coef_,
                            df_var_preditoras.columns,
                            columns=['Coeficiente'])
print(f"Coeficientes: {coeficientes}")

## calculo dos erros
y_predicao = modelo_reg_multi.predict(x_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_predicao))
print('MSE:', metrics.mean_squared_error(y_test, y_predicao))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predicao)))
## remoção da variável 'age' aumenta o erro
## remoção da variável 'bmi' aumenta um pouco o erro
## remoção da variável 'children' aumenta um pouco o erro médio, mas
## aumenta bastante o erro RMSE
