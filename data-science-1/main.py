#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[100]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[ ]:





# ## Parte 1

# ### _Setup_ da parte 1

# In[101]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ### Inicie sua análise a partir da parte 1 a partir daqui

# In[102]:


dataframe.head()


# In[103]:


desc = dataframe.describe()
desc


# In[104]:


#fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=False)

#fig.suptitle('Histogramas Normal x Binomial')
#axs[0].hist(dataframe.normal,bins=50)
#axs[1].hist(dataframe.binomial,bins=50)


# In[105]:


#sns.distplot(dataframe['normal'])
#sns.distplot(dataframe['binomial'])


# ### Solução Q1:

# In[106]:


lista1=[]
for i in range(4,7):
    dif = round(desc.iloc[i,0]-desc.iloc[i,1],3)
    lista1.append(dif)
questao1 = (lista1[0],lista1[1],lista1[2])
questao1


# ### Solução Q1 alternativa:

# In[107]:


qt = dataframe.quantile([0.25,0.5,0.75])
q001 = tuple((qt['normal']-qt['binomial']).round(3))
q001


# ### Solução Q2:

# In[108]:


import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


inf = dataframe.normal.mean()-dataframe.normal.std()
sup = dataframe.normal.mean()+dataframe.normal.std()
ecdf = ECDF(dataframe.normal)
questao2 = (ecdf(sup)-ecdf(inf)).round(3)
questao2


# ### Solução Q2 alternativa:

# In[109]:


mean = dataframe['normal'].mean()
std = dataframe['normal'].std()
x1_s1 = mean - std 
x2_s1 = mean + std

question2 = float((ECDF(dataframe['normal'])(x2_s1) - ECDF(dataframe['normal'])(x1_s1)).round(3))
question2


# ### Solução Q3:

# In[110]:


diff_m = round(dataframe.binomial.mean()-dataframe.normal.mean(),3)
diff_v = round(pow(dataframe.binomial.std(),2)-pow(dataframe.normal.std(),2),3)
questao3 = (diff_m,diff_v)
questao3


# ### Solução Q3 alternativa:

# In[111]:


df = dataframe.agg(['mean', 'std'])
df['dif'] = df['binomial'] - df['normal']

m = df['dif']['mean'].round(3)
v = df['dif']['std'].round(3)
question3=(m,v)
question3


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[112]:


def q1():
    quant = dataframe.quantile([0.25, 0.5, 0.75])
    return tuple((quant['normal'] - quant['binomial']).round(3))


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[126]:


def q2():
    return questao2


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[127]:


def q3():
    return questao3


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[115]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[116]:


stars.head()


# In[117]:


stars.describe()


# ### Solução Questão 4:

# In[118]:


false_pulsar_mean_profile = stars[stars['target'] == 0]['mean_profile']
false_pulsar_mean_profile_standardized = (false_pulsar_mean_profile - false_pulsar_mean_profile.mean())/(false_pulsar_mean_profile.std())

quantis = sct.norm.ppf([0.8, 0.9, 0.95])
fecdf = ECDF(false_pulsar_mean_profile_standardized)
prob = [fecdf(q).round(3) for q in quantis]
prob


# ### Solução Questão 5: 

# In[119]:


normal = sct.norm.ppf([0.25,0.50,0.75])
std = false_pulsar_mean_profile_standardized.quantile([0.25,0.50,0.75])
questao5 = tuple((std.values - normal).round(3))
questao5


# In[120]:


#sct.probplot(false_pulsar_mean_profile_standardized, dist="norm", plot=plt)
#plt.title("Gráfico Q-Q")
#plt.show()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[121]:


def q4():
    prob = [fecdf(q).round(3) for q in quantis]
    return tuple(prob)


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[122]:


def q5():
    normal = sct.norm.ppf([0.25,0.50,0.75])
    std = false_pulsar_mean_profile_standardized.quantile([0.25,0.50,0.75])
    return tuple((std.values - normal).round(3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
