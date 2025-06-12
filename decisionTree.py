#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:13:48 2025

@author: clovis
"""

import os

os.chdir('/home/clovis/dSinc/Aulas/machine_learning/datasets')



import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df_imoveis = pd.read_csv('imoveis.csv')

# # Dados da tabela (Área, Quartos, Preço)
# dados = [
#     [80, 2, 300000], [90, 3, 350000], [100, 3, 400000],
#     [120, 4, 480000], [150, 4, 600000], [70, 1, 250000],
#     [85, 2, 320000], [95, 3, 370000], [110, 3, 420000],
#     [130, 4, 500000], [140, 4, 550000], [160, 5, 650000],
#     [65, 1, 230000], [75, 2, 280000], [105, 3, 390000]
# ]

# # Convertendo para arrays numpy
# dados = np.array(dados)
# x1 = dados[:, 0]  # Área
# x2 = dados[:, 1]  # Quartos
# y = dados[:, 2]   # Preço

# Criando o gráfico de dispersão com cores baseadas no preço
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df_imoveis['Area'], df_imoveis['Quartos'], c=df_imoveis['Preco'], cmap='viridis', s=100, alpha=0.8, edgecolor='k'
)
plt.axvline(x=115, color='red', linestyle=':', linewidth=1.5, label=f'Quartos ≤ 2.5')
plt.axhline(y=2.5, color='red', linestyle=':', linewidth=1.5, label=f'Quartos ≤ 110')

# Adicionando barra de cores (legenda)
cbar = plt.colorbar(scatter)
cbar.set_label('Preço do Imóvel (R$)', fontsize=12)

# Personalizando o gráfico
plt.title('Relação entre Área, Número de Quartos e Preço', fontsize=14)
plt.xlabel('Área (m²)', fontsize=12)
plt.ylabel('Número de Quartos', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Ajustando os ticks do eixo y (quartos são inteiros)
plt.yticks(np.arange(1, 6, 1))

# Mostrando o gráfico
plt.show()


mxd = 2
# Criando e treinando a árvore de regressão
modelo = DecisionTreeRegressor(
    max_depth=mxd,    # Profundidade máxima para evitar overfitting
    random_state=42
)
modelo.fit(df_imoveis[['Area', 'Quartos']], df_imoveis['Preco'])
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 12))
# plot_tree(modelo, feature_names=['Area', 'Quartos'], filled=True)    
plot_tree(modelo, 
          feature_names=['Area', 'Quartos'], 
          filled=True, 
          rounded=True,
          fontsize=6,
          proportion=True,
          precision=2)

plt.show()

from sklearn.tree import _tree

def get_decision_boundaries(tree, feature_names):
    boundaries = {}
    tree_ = tree.tree_
    
    for i in range(tree_.node_count):
        if tree_.feature[i] != _tree.TREE_UNDEFINED:  # Se não for nó folha
            feature_name = feature_names[tree_.feature[i]]
            threshold = tree_.threshold[i]
            if feature_name not in boundaries:
                boundaries[feature_name] = []
            boundaries[feature_name].append(threshold)
    
    return boundaries

# Obter os limiares de divisão
boundaries = get_decision_boundaries(modelo, ['Area', 'Quartos'])
print(boundaries)


import numpy as np

plt.figure(figsize=(10, 6))

# 1. Scatter plot dos dados
scatter = plt.scatter(
    df_imoveis['Area'], df_imoveis['Quartos'], c=df_imoveis['Preco'], cmap='viridis', s=100, alpha=0.8, edgecolor='k'
)
plt.colorbar(label='Preço (R$)')

# 2. Linhas de divisão (extraídas da árvore)
# Linhas verticais (divisões por Área)
for threshold in boundaries['Area']:
    plt.axvline(x=threshold, color='red', linestyle=':', linewidth=1.5, label=f'Área ≤ {threshold}')

# Linhas horizontais (divisões por Quartos)
for threshold in boundaries['Quartos']:
    plt.axhline(y=threshold, color='red', linestyle=':', linewidth=1.5, label=f'Quartos ≤ {threshold}')

# Configurações do gráfico
plt.xlabel('Área (m²)')
plt.ylabel('Número de Quartos')
plt.title(f'Divisões da Árvore de Decisão (Profundidade={mxd})')
# plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()    
plt.show()