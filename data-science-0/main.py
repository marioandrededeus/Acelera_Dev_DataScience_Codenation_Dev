#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[46]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# In[47]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[ ]:





# ## VISUALIZAÇÃO DO DATASET

# In[48]:


df=pd.read_csv('black_friday.csv')
df.head()


# ## QUESTÃO 1: NÚMERO DE OBSERVAÇÕES, NÚMERO DE COLUNAS

# In[49]:


df.shape


# In[50]:


df.info()


# ## VERIFICAÇÃO  DOS TIPOS DE ENTRADAS DE CADA FEATURE

# In[51]:


col = df.columns

for i in col:
    print(f'{i}\n {df[i].value_counts()}\n')


# ## ADEQUAÇÃO DO "TIPO" DE CADA FEATURE (DTYPE)

# In[52]:


df.dtypes


# In[53]:


df2 = df.copy()
df2['Occupation'] = df2['Occupation'].astype('object')
df2['Stay_In_Current_City_Years'] = df2['Stay_In_Current_City_Years'].astype('object')
df2['Marital_Status'] = df2['Marital_Status'].astype('object')
df2['Product_Category_1'] = df2['Product_Category_1'].astype('object')
df2['Product_Category_2'] = df2['Product_Category_2'].astype('object')
df2['Product_Category_3'] = df2['Product_Category_3'].astype('object')
df2['Purchase'] = df2['Purchase'].astype('float64')
df2.dtypes


# ## QUESTÃO 2: NÚMERO DE MULHERES COM IDADE ENTRE 26 E 35

# In[54]:


df_female = df['Gender']=='F'
df_26_35 = df['Age']=='26-35'
df_q2 = df[df_female & df_26_35]
df_q2.shape[0]


# ## QUESTÃO 3: NÚMERO DE USUÁRIOS ÚNICOS

# In[55]:


df['User_ID'].nunique()


# ## QUESTÃO 4: NÚMERO DE TIPOS DE DADOS DIFERENTES NO DATASET

# In[56]:


df.dtypes.nunique()


# ## QUESTÃO 5: PORCENTAGEM DE OBSERVAÇÕES COM PELO MENOS UM VALOR NULO

# In[57]:


df_prod2_notnull = df['Product_Category_2'].notna()
df_prod3_notnull = df['Product_Category_3'].notna()
df_prod2_3_notnull = df[df_prod2_notnull & df_prod3_notnull]

#Qtd observações com pelo menos 1 valor nulo:
obs_nulos = len (df) - len(df_prod2_3_notnull)

#Porcentagem de observações com pelo menos 1 valor nulo:
percent_nulos = obs_nulos / len(df)
percent_nulos


# ## QUESTÃO 6: NÚMERO DE NULL NA VARIÁVEL COM MAIOR NÚMERO DE NULL

# In[58]:


df['Product_Category_3'].isnull().sum()


# # QUESTÃO 7: VALOR MAIS FREQUENTE (NOTNULL) NA VARIÁVEL Product_Category_3 

# In[59]:


df['Product_Category_3'].value_counts()


# ## QUESTÃO 8: MÉDIA DA VARIÁVEL PURCHASE APÓS UMA NORMALIZAÇÃO

# In[60]:


purchase = np.array(df.Purchase)
xmin=df['Purchase'].min()
xmax=df['Purchase'].max()

x_normalizado=[]

for x in purchase:
    xn = (x-xmin)/(xmax-xmin)
    x_normalizado.append(xn)
purchase_norm = pd.DataFrame(x_normalizado)
purchase_norm.describe()


# ## QUESTÃO 9: Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[61]:


purchase = np.array(df.Purchase)
media=df['Purchase'].mean()
desvio=df['Purchase'].std()

x_std=[]

for x in purchase:
    xs = (x-media)/desvio
    x_std.append(xs)
df['purchase_std'] = pd.DataFrame(x_std)
df.head()


# In[62]:


p_maiormenos1 = df['purchase_std']>-1
p_menorigual1 = df['purchase_std']<=1
questao9 = df[p_maiormenos1 & p_menorigual1]
len(questao9['purchase_std'])


# ## QUESTÃO 10: Podemos afirmar que se uma observação é null em Product_Category_2 ela também o é em Product_Category_3 ?

# In[63]:


prod2_null = df['Product_Category_2'].isnull()
prod3_notnull = df['Product_Category_3'].notnull()
len(df[prod2_null & prod3_notnull])


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[75]:


def q1():
    return (537577, 12)


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[65]:


def q2():
    return 49348


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[66]:


def q3():
    return 891


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[67]:


def q4():
    return 3


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[68]:


def q5():
    return 0.6944102891306734


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[69]:


def q6():
    return 373299


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[70]:


def q7():
    return 16


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[71]:


def q8():
    return 0.384794


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[72]:


def q9():
    return 348631


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[73]:


def q10():
    return True

