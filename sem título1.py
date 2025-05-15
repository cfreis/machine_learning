#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:41:36 2025

@author: clovis
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import os
os.chdir('/home/clovis/dSinc/Aulas/machine_learning/')



# Gerar dados com resíduos normais
np.random.seed(42)
X = np.linspace(0, 10, 100)
# Gerando dados com resíduos normais
y_norm = 2 * X + 1 + np.random.normal(0, 1, 100) 

# Ajustar modelo de regressão linear
# Agora utilizando a função OLS da biblioteca statsmodels
X_with_const = sm.add_constant(X)
model_norm = sm.OLS(y_norm, X_with_const).fit()
residuos_norm = model_norm.resid

# Gerar dados com resíduos não-normais (exponenciais)
y_N_norm = 2 * X + 1 + np.random.exponential(1, 100) - 1  

# Ajustar modelo de regressão linear
model_N_norm = sm.OLS(y_N_norm, X_with_const).fit()
residuos_N_norm = model_N_norm.resid


import regression_tests as rt
rt.diagnostic_plots(model_N_norm,n_cols=2,plots=['regressao','regressao','cook','cook','buga','residuos'],save_path='/home/clovis/Imagens/fig01.png')
#['residuos', 'qq', 'hist', 'scale', 'leverage']
# rt.diagnostic_plots(model_N_norm,['KS'])

# rt.diagnostic_plots(model_N_norm)
# rt.KS_test(model_N_norm,)
# rt.KS_test(model_norm)

# rt.shapiro_test(model_N_norm)
# rt.anderson_tests(model_norm, vprint=True, rigor=0)

# resultados = rt.test_residues(
#     model=model_norm,
#     tests=['KS', 'Shapiro', 'Anderson'],  # Testes a serem realizados
#     alpha=0.05,                          # Nível de significância
#     rigor=2,                             # Nível para Anderson-Darling (2 = 5%)
#     vprint=False                         # Não imprimir detalhes
# )

# #display(resultados)
# print(resultados)


# model_norm.summary()
# import statsmodels.api as sm

# # X_stat = sm.add_constant(X_h)
# # regsummary = sm.OLS(y_com_h, X_stat).fit()
# # print(regsummary.summary())


# '''







# '''