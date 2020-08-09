#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from loguru import logger


# In[2]:


fifa = pd.read_csv("fifa.csv")
fifa.columns.shape


# In[3]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[4]:


fifa.head()
fifa.columns.shape


# In[5]:


fifa.info()


# In[6]:


fifa[fifa['Crossing'].isnull()].shape


# In[7]:


fifa2 = fifa.copy().dropna(axis=0)
fifa2.shape


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[8]:


pca = PCA()
pca.fit(fifa2)
pca_explain = pca.explained_variance_ratio_.round(3)
pca_explain


# In[9]:


def q1():
    pca = PCA().fit(fifa2)
    pca_explain = pca.explained_variance_ratio_
    return pca_explain[0].round(3)


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[10]:


cont,vt = 0,0
for var in pca_explain:
    vt += var
    cont +=1
    print(vt.round(2),cont)
    if vt.round(2) >=0.95:
        break
#print(cont)


# In[11]:


PCA(n_components=0.95).fit_transform(fifa2).shape[1]


# In[12]:


def q2():
    return PCA(n_components=0.95).fit_transform(fifa2).shape[1]


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[13]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[14]:


pca = PCA(n_components=2).fit(fifa2)
coordenadas1 = pca.components_.T #coordenadas de cada uma das 37 variaveis
coordenadas2 = np.dot(x,coordenadas1) #produto entre duas arrays
print(tuple(coordenadas2.round(3)))
pca_fifa2 = pca.transform(fifa2)
print('Original_shape: ',fifa2.shape)
print('Transformed_shape: ',pca_fifa2.shape)


# In[15]:


def q3():
    return tuple(coordenadas2.round(3))


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[16]:


#Separating out the features
X = fifa2[fifa2.columns.difference(['Overall'])]

#Separating out the target
y = fifa2['Overall']

#Define RFE with Linear Regressor as estimator 
rfe = RFE(LinearRegression(), n_features_to_select = 5)

#Fit RFE
rfe = rfe.fit(X, y)

#Create a dataframe for summarize the results
df = pd.DataFrame({'Variaveis': X.columns, 'Selected': rfe.support_, 'Rank': rfe.ranking_})
df


# In[17]:


list(df.Variaveis[df['Selected']==True])


# In[18]:


def q4():
    return list(df.Variaveis[df['Selected']==True])

