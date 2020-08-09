#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[76]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)


# In[77]:


countries = pd.read_csv("countries.csv",decimal=",")


# In[78]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head()


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[79]:


countries.info()


# In[80]:


countries['Country'] = countries['Country'].str.rstrip()
countries['Country'] = countries['Country'].str.lstrip()
countries['Region'] = countries['Region'].str.rstrip()
countries['Region'] = countries['Region'].str.lstrip()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[97]:


def q1():
    regions = countries['Region'].sort_values().unique()
    return list(regions)
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[82]:


discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")

discretizer.fit(countries[["Pop_density"]])

score_bins = discretizer.transform(countries[["Pop_density"]]).astype(int)

sns.distplot(score_bins)

print(score_bins.dtype)
print(np.sum(score_bins==9).astype(int))


# In[83]:


def q2():
    return np.sum(score_bins==9).astype(int)
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[98]:


def q3():
    region_unique = len(countries.Region.unique())
    climate_unique = len(countries.Climate.unique())
    return region_unique + climate_unique
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[85]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[95]:


def q4():
    transformed_countries = countries.copy()
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("standard", StandardScaler())])
        
    pipe.fit(transformed_countries.iloc[:,2:])
    data_transformed = pipe.transform([test_country[2:]])
    return round(data_transformed[0, transformed_countries.columns.get_loc("Arable") - 2],3)
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[89]:


sns.boxplot(countries.Net_migration, orient="vertical")


# In[94]:


def q5():
    outlier_net_migration = countries.Net_migration.copy()
    q1 = outlier_net_migration.quantile(0.25)
    q3 = outlier_net_migration.quantile(0.75)
    iqr = q3 - q1

    min_lim = q1 - 1.5 * iqr
    max_lim = q3 + 1.5 * iqr

    outliers_abaixo = len(outlier_net_migration[outlier_net_migration< min_lim])
    outliers_acima = len(outlier_net_migration[outlier_net_migration> max_lim])
    return outliers_abaixo,outliers_acima,False
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[92]:


def q6():
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    # \"Vectorizando\" o data set"
    counter = CountVectorizer()
    freq = counter.fit_transform(newsgroup.data)
    # Recebendo o vocabulário\n",
    words = dict(counter.vocabulary_.items())
    # Retornando a soma.
    return(int(freq[:,words['phone']].sum()))
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[93]:


def q7():
    # Criando o data set
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    #Vectorizando o data set
    Tfid = TfidfVectorizer()
    freq = Tfid.fit_transform(newsgroup.data)
    # Retornando o idf
    return(round(float(freq[:,Tfid.vocabulary_['phone']].sum()),3))
q7()

