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
rt.diagnostic_plots(model_N_norm,n_cols=2)
#['residuos', 'qq', 'hist', 'scale', 'leverage']
rt.diagnostic_plots(model_N_norm,['KS'])

rt.diagnostic_plots(model_N_norm)
rt.KS_test(model_N_norm, plot=False)
rt.KS_test(model_norm)

rt.shapiro_test(model_N_norm)
rt.anderson_tests(model_norm, vprint=True, rigor=0)

resultados = rt.test_residues(
    model=model_norm,
    tests=['KS', 'Shapiro', 'Anderson'],  # Testes a serem realizados
    alpha=0.05,                          # Nível de significância
    rigor=2,                             # Nível para Anderson-Darling (2 = 5%)
    plot=True,                           # Mostrar gráficos
    vprint=False                         # Não imprimir detalhes
)

display(resultados)
print(resultados)

'''
Análise do Gráfico "Resíduos vs Valores Ajustados"
O gráfico de Resíduos vs Valores Ajustados é um dos diagnósticos mais importantes para verificar as suposições de um modelo de regressão linear. Ele ajuda a identificar:

Homocedasticidade (variância constante dos resíduos)

Padrões não-lineares nos resíduos

Outliers ou pontos influentes

Como Interpretar o Gráfico
1. Padrão Ideal (Homocedasticidade)
O que esperar:
Os resíduos devem estar aleatoriamente dispersos em torno da linha horizontal em zero, sem padrões claros e com dispersão constante.

python
# Exemplo de código para gerar um padrão ideal (simulado)
np.random.seed(42)
residuos_ideais = np.random.normal(0, 1, 100)
fitted_ideais = np.linspace(0, 10, 100)
sns.residplot(x=fitted_ideais, y=residuos_ideais, lowess=True)
Homocedasticidade
(Resíduos distribuídos aleatoriamente, sem tendências)

2. Heterocedasticidade (Problema Comum)
O que observar:

Formato de "funil" (a dispersão aumenta/diminui com os valores ajustados).

Indica que a variância dos erros não é constante.

python
# Exemplo de heterocedasticidade (variância aumenta com X)
residuos_hetero = np.random.normal(0, fitted_ideais, 100)
sns.residplot(x=fitted_ideais, y=residuos_hetero, lowess=True)
Heterocedasticidade
Solução: Transformar a variável dependente (ex.: log(y)) ou usar modelos robustos (WLS).

3. Não-Linearidade
O que observar:

Padrões curvos ou sistemáticos (ex.: forma de "U" ou "onda").

Indica que o modelo não capturou uma relação não-linear.

python
# Exemplo de não-linearidade (resíduos em forma de U)
residuos_naolinear = 0.5 * (fitted_ideais - 5)**2 + np.random.normal(0, 1, 100)
sns.residplot(x=fitted_ideais, y=residuos_naolinear, lowess=True)
Não-linearidade
Solução: Adicionar termos polinomiais (ex.: X²) ou usar modelos não-lineares.

4. Outliers e Pontos Influentes
O que observar:

Pontos muito distantes da linha zero (acima de ±3 desvios padrão).

Podem distorcer o modelo.

python
# Exemplo com outliers
residuos_outlier = np.random.normal(0, 1, 100)
residuos_outlier[95] = 10  # Outlier artificial
sns.residplot(x=fitted_ideais, y=residuos_outlier, lowess=True)
Outliers
Solução: Verificar se são erros de medição ou usar métodos robustos (ex.: RANSAC).

Linha LOWESS no Gráfico
A linha vermelha (lowess=True) mostra a tendência local dos resíduos:

Se seguir horizontalmente próximo a zero: bom sinal.

Se for curva ou inclinada: indica problemas (não-linearidade ou heterocedasticidade).




Análise do Gráfico Q-Q (Quantil-Quantil)
O gráfico Q-Q (Quantil-Quantil) é uma ferramenta essencial para verificar se os resíduos de um modelo de regressão seguem uma distribuição normal, uma das premissas fundamentais da regressão linear. Aqui está como interpretá-lo:

1. O que o Gráfico Q-Q Mostra?
Eixo X: Quantis teóricos da distribuição normal (o que esperar se os resíduos fossem perfeitamente normais).

Eixo Y: Quantis dos resíduos observados (seus dados).

Linha de referência (linha 's'): Representa a distribuição normal ideal.

Objetivo:
Verificar se os pontos (resíduos) se alinham à linha teórica. Se sim, os resíduos são normais; se não, há desvios.

2. Padrões Típicos e Suas Interpretações
✅ Caso Ideal (Normalidade)
O que ver:
Os pontos se alinham quase perfeitamente sobre a linha reta.

python
from statsmodels.graphics.gofplots import qqplot
import numpy as np

# Gerar resíduos normais (exemplo ideal)
residuos_normais = np.random.normal(0, 1, 100)
qqplot(residuos_normais, line='s', color='blue')  # 's' = linha teórica
plt.title("Q-Q Plot (Normalidade)")
plt.show()
Q-Q Plot Ideal
(Pontos próximos à linha, sem desvios sistemáticos)

❌ Problemas Comuns
(A) Caudas Pesadas (Kurtose)
O que ver:
Pontos nas extremidades (caudas) acima ou abaixo da linha.

Cauda acima da linha: Resíduos têm mais valores extremos que o esperado (cauda pesada).

Cauda abaixo da linha: Resíduos têm menos valores extremos (cauda leve).

Exemplo (cauda pesada):

python
residuos_cauda_pesada = np.random.standard_t(df=3, size=100)  # Distribuição t (caudas pesadas)
qqplot(residuos_cauda_pesada, line='s')
plt.title("Q-Q Plot (Caudas Pesadas)")
Caudas Pesadas
Solução:

Transformar y (ex.: log(y), Box-Cox).

Usar modelos robustos a outliers (ex.: Regressão Quantílica).

(B) Assimetria (Skewness)
O que ver:

Curva para cima no início/fim: Assimetria positiva (resíduos inclinados para a direita).

Curva para baixo no início/fim: Assimetria negativa (resíduos inclinados para a esquerda).

Exemplo (assimetria positiva):

python
residuos_assimetricos = np.random.exponential(scale=1, size=100)  # Assimetria positiva
qqplot(residuos_assimetricos, line='s')
plt.title("Q-Q Plot (Assimetria Positiva)")
Assimetria
Solução:

Transformar y (ex.: np.sqrt(y), np.log(y)).

(C) Desvios Sistemáticos (Não-Normalidade)
O que ver:
Padrão em "S" ou "U", indicando que a distribuição dos resíduos não é normal.

Exemplo (padrão em S):

python
residuos_nao_normais = np.random.uniform(low=-2, high=2, size=100)  # Distribuição uniforme
qqplot(residuos_nao_normais, line='s')
plt.title("Q-Q Plot (Não-Normalidade)")
Não-Normalidade
Solução:

Verificar outliers ou erros de medição.

Considerar modelos não-lineares ou não-paramétricos.

3. Como Implementar no Código Original
No seu código, o Q-Q Plot é gerado por:

python
from statsmodels.graphics.gofplots import qqplot

qqplot(residuos, line='s', ax=plt.gca())  # 's' = linha teórica
plt.title('Q-Q Plot')
line='s': Adiciona a linha de referência normal.

ax=plt.gca(): Usa o subplot atual para o gráfico.

4. Resumo das Interpretações
Padrão no Q-Q Plot	Problema	Solução
Pontos alinhados à linha	Normalidade (OK)	Nenhuma ação.
Caudas acima/abaixo	Kurtose (caudas pesadas)	Transformar y (ex.: log(y)).
Curva para cima/baixo	Assimetria	Usar np.sqrt(y) ou np.log1p(y).
Padrão em "S" ou "U"	Não-normalidade	Verificar outliers ou usar modelos robustos.
5. Dicas Práticas
Combine com outros gráficos:

Se o Q-Q Plot mostrar não-normalidade, verifique também o histograma dos resíduos (3º gráfico do seu painel).

Testes estatísticos complementares:

Shapiro-Wilk (scipy.stats.shapiro): Para amostras pequenas (< 50).

Kolmogorov-Smirnov (scipy.stats.kstest): Para amostras maiores.

Exemplo de teste de normalidade:

python
from scipy.stats import shapiro
stat, p_valor = shapiro(residuos)
print(f"Shapiro-Wilk p-valor: {p_valor:.4f}")
if p_valor > 0.05:
    print("Normalidade (não rejeita H0)")
else:
    print("Não-normalidade (rejeita H0)")
Conclusão
O Q-Q Plot é o melhor gráfico para checar normalidade dos resíduos. Se houver desvios:

Transforme a variável y (log, Box-Cox).

Use modelos robustos se a não-normalidade persistir.



Análise do Gráfico Scale-Location (Scale-Location Plot)
O gráfico Scale-Location (também chamado de Spread-Location) é usado para verificar a homocedasticidade (variância constante dos resíduos) de forma mais clara que o gráfico de resíduos vs. ajustados. Ele plota:

Eixo X: Valores ajustados (
y
^
y
^
​
 )

Eixo Y: Raiz quadrada dos resíduos padronizados em valor absoluto (
∣
res
ı
ˊ
duos padronizados
∣
∣res 
ı
ˊ
 duos padronizados∣
​
 )

Como Interpretar o Gráfico Scale-Location
1. Padrão Ideal (Homocedasticidade)
O que esperar:

Uma linha horizontal (linha LOWESS) sem tendência clara.

Dispersão uniforme dos pontos ao longo de todos os valores ajustados.

python
# Exemplo de código para um padrão ideal
plt.figure(figsize=(8, 4))
sns.regplot(x=fitted, y=np.sqrt(np.abs(residuos / residuos.std())), 
            lowess=True, 
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'})
plt.title("Scale-Location (Homocedasticidade)")
Homocedasticidade
(Linha vermelha aproximadamente plana, sem padrões claros)

2. Heterocedasticidade (Problema)
O que observar:

Linha LOWESS ascendente ou descendente.

Dispersão dos pontos aumenta/diminui com os valores ajustados.

python
# Exemplo de heterocedasticidade (variância aumenta com X)
residuos_hetero = np.random.normal(0, fitted, 100)
sns.regplot(x=fitted, y=np.sqrt(np.abs(residuos_hetero / residuos_hetero.std())), 
            lowess=True)
Heterocedasticidade
Solução:

Transformar a variável dependente (ex.: log(y)).

Usar Mínimos Quadrados Ponderados (WLS).

3. Padrões Não-Lineares
O que observar:

Linha LOWESS com curvas ou ondulações.

Indica que o modelo não capturou uma relação não-linear.

python
# Exemplo de não-linearidade
residuos_naolinear = 0.1 * (fitted - 5)**2 + np.random.normal(0, 1, 100)
sns.regplot(x=fitted, y=np.sqrt(np.abs(residuos_naolinear / residuos_naolinear.std())), 
            lowess=True)
Não-linearidade
Solução:

Adicionar termos polinomiais (ex.: X², X³).

Usar modelos não-lineares (ex.: regressão polinomial).

Por Que Usar Resíduos Padronizados?
A fórmula 
∣
res
ı
ˊ
duos
/
desvio padr
a
˜
o
∣
∣res 
ı
ˊ
 duos/desvio padr 
a
˜
 o∣
​
  padroniza os resíduos para:

Facilitar a comparação entre modelos.

Identificar melhor a heterocedasticidade.

Exemplo no Código Original
No seu código, o gráfico é gerado por:

python
residuos_norm = np.sqrt(np.abs(residuos / residuos.std()))
sns.regplot(x=fitted, y=residuos_norm, lowess=True, 
           scatter_kws={'alpha': 0.6},
           line_kws={'color': 'red', 'lw': 1})
Resumo dos Problemas e Soluções
Padrão no Gráfico	Problema	Solução
Linha horizontal	Homocedasticidade (OK)	Nenhuma ação necessária.
Linha ascendente/descendente	Heterocedasticidade	Transformar y (ex.: log(y)), WLS.
Linha curva	Não-linearidade	Adicionar termos polinomiais.
Comparação com Outros Gráficos de Diagnóstico
Resíduos vs Ajustados: Mostra heterocedasticidade e não-linearidade, mas menos claro.

Scale-Location: Foca na variância dos resíduos (mais sensível para heterocedasticidade).

Q-Q Plot: Verifica normalidade, não variância.

Dica Prática
Se a linha LOWESS no Scale-Location não for horizontal:

Transforme y:

python
model_log = sm.OLS(np.log(y), X).fit()
diagnostic_plots(model_log)  # Verifique novamente
Use WLS se a transformação não resolver:

python
model_wls = sm.WLS(y, X, weights=1/fitted).fit()





5. Comparação com Outros Testes
Teste	Uso Recomendado	Vantagem
Anderson-Darling	Amostras grandes (>50)	Sensível a caudas da distribuição
Shapiro-Wilk	Amostras pequenas (<50)	Melhor para normalidade em pequenas amostras
Kolmogorov-Smirnov	Qualquer tamanho	Testa qualquer distribuição, mas menos potente
Exemplo com Shapiro-Wilk:

python
from scipy.stats import shapiro
stat, p = shapiro(dados)
print(f"Shapiro-Wilk p-valor: {p:.4f}")
6. Exemplo Completo com Gráfico
python
Saída:
Histograma com teste A-D

Quando Usar o Anderson-Darling?
Para verificar normalidade em amostras grandes.

Quando você precisa de maior sensibilidade nas caudas da distribuição.

Alternativa ao Shapiro-Wilk para 
n
>
50
n>50.

Se precisar testar outras distribuições ou comparar com outros testes, posso ajudar com exemplos específicos!
'''