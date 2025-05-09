import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

sns.set_style("whitegrid")


def KS_test(model, plot = True, alpha=0.05):
    '''
    Realiza o teste KS, comparando com a distribuição normal padrão e
        cria os gráficos correspondentes.
    Parâmetros:
        residuos - numpy.ndarray contendo os resíduos da regressão linear
        plot - imprime o gráfico KS - default=True
        alpha - valor para comparação com o p-valor - default=0.05
    '''
    residuos, fitted = extrai_dados(model)
    KS_statistic, KS_p_value = stats.kstest(residuos, 'norm')  # 'norm' = distribuição normal padrão
    
    result = f'Kolmogorov-Smirnov;{KS_statistic:.4f};{KS_p_value:.4f};H0=Distr Normal -> '
    
    # Interpretação
    if KS_p_value > alpha:
        result = f'{result}Não rejeitar'
    else:
        result = f'{result}Rejeitar'
        
    if plot:
        # Gerar dados
        # Calcular CDF empírica e teórica
        x = np.linspace(-4, 4, 1000)
        cdf_empirica = np.array([np.sum(residuos <= xx)/len(residuos) for xx in x])
        cdf_teorica = stats.norm.cdf(x)
        
        # Encontrar o ponto de máxima diferença
        idx_max = np.argmax(np.abs(cdf_empirica - cdf_teorica))
        
        # Plotar
        plt.figure(figsize=(10, 6))
        plt.plot(x, cdf_teorica, label='CDF Teórica N(0,1)')
        plt.plot(x, cdf_empirica, label='CDF Empírica')
        plt.vlines(x=x[idx_max], ymin=cdf_teorica[idx_max], ymax=cdf_empirica[idx_max], 
                   color='red', linestyle='--', 
                   label=f'Diferença máxima: {KS_statistic:.3f}')
        plt.title('Teste de Kolmogorov-Smirnov')
        plt.xlabel('Valores')
        plt.ylabel('Probabilidade Acumulada')
        plt.legend()
        plt.show()
        
    return(result)



def shapiro_test(model, alpha = 0.05):
    '''
    Realiza o teste de normalidade de Shapiro-Wilk
    Parâmetros:
        residuos - numpy.ndarray contendo os resíduos da regressão linear
        alpha - valor para comparação com o p-valor - default=0.05
    '''
    residuos, fitted = extrai_dados(model)

    sh_statistic, sh_p_value = stats.shapiro(residuos)
    
    result = f'Shapiro-Wilk;{sh_statistic:.4f};{sh_p_value:.4f};H0=Distr Normal -> '
    
    # Interpretação
    if sh_p_value > alpha:
        result = f'{result}Não rejeitar'
    else:
        result = f'{result}Rejeitar'

    return(result)

def extrai_dados(model):
    residuos = model.resid
    fitted = model.fittedvalues

    return(residuos, fitted)

def diagnostic_plots(model, plots=['residuos', 'qq', 'hist', 'scale', 'leverage'], n_cols=2, figsize=(0, 0)):
    """
    Cria subplots dinâmicos com os gráficos de diagnóstico selecionados.
    
    Parâmetros:
    -----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo de regressão ajustado
    plots : list, optional
        Lista dos gráficos a incluir (opções: 'residuos', 'qq', 'hist', 'scale', 'leverage')
    n_cols : inteiro, opcional
        Túmero de colunas de gráficos no sub-plot. Default=2
    figsize : tuple, optional
        Tamanho da figura (largura, altura)
    """
    # Dados do modelo
    fitted = model.fittedvalues
    residuos = model.resid
    influence = OLSInfluence(model)
    leverage = influence.hat_matrix_diag
    cooks_distance = influence.cooks_distance[0]
    
    # Mapeamento das funções de plot
    plot_functions = {
        'residuos': (plot_residuos_vs_ajustados, (fitted, residuos)),
        'qq': (plot_qq, (residuos,)),
        'hist': (plot_hist_residuos, (residuos,)),
        'scale': (plot_scale_location, (fitted, residuos)),
        'leverage': (plot_leverage, (residuos, leverage, cooks_distance))
    }
    
    # Filtra apenas os plots solicitados e existentes
    valid_plots = [p for p in plots if p in plot_functions]
    n_plots = len(valid_plots)
    
    if n_plots == 0:
        raise ValueError("Nenhum gráfico válido selecionado. Opções: 'residuos', 'qq', 'hist', 'scale', 'leverage'")
    
    # Configura layout dos subplots
    n_cols = min(n_cols, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    if figsize[0] == 0:
        figsize = (n_cols*5,n_rows*4)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])  # Garante que axes seja sempre um array
        
    # Achata o array de eixos para facilitar iteração
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Gera cada gráfico
    for i, plot_name in enumerate(valid_plots):
        ax = axes_flat[i]
        plot_func, args = plot_functions[plot_name]
        plot_func(*args, ax=ax)
    
    # Remove eixos vazios
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    plt.show()



   
def plot_residuos_vs_ajustados(fitted, residuos, ax):
    """Gráfico de Resíduos vs Valores Ajustados"""
    sns.residplot(x=fitted, y=residuos, lowess=True,
                 scatter_kws={'alpha': 0.6},
                 line_kws={'color': 'red', 'lw': 1}, ax=ax)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title('Resíduos vs Ajustados')
    ax.set_xlabel('Valores Ajustados')
    ax.set_ylabel('Resíduos')

def plot_qq(residuos, ax):
    """Gráfico Q-Q para normalidade"""
    qqplot(residuos, line='s', ax=ax)
    ax.set_title('Q-Q Plot')
    ax.get_lines()[0].set_markersize(4.0)  # Ajusta tamanho dos pontos


def plot_hist_residuos(residuos, ax):
    """Histograma dos Resíduos"""
    sns.histplot(residuos, kde=True, ax=ax)
    ax.set_title('Distribuição dos Resíduos')
    ax.set_xlabel('Resíduos')
   
def plot_scale_location(fitted, residuos, ax):
    """Gráfico Scale-Location"""
    residuos_norm = np.sqrt(np.abs(residuos / residuos.std()))
    sns.regplot(x=fitted, y=residuos_norm, lowess=True,
               scatter_kws={'alpha': 0.6},
               line_kws={'color': 'red', 'lw': 1}, ax=ax)
    ax.set_title('Scale-Location')
    ax.set_xlabel('Valores Ajustados')
    ax.set_ylabel('√(|Resíduos Padronizados|)')

def plot_leverage(residuos, leverage, cooks_distance, ax):
    """Gráfico de Leverage"""
    sns.scatterplot(x=leverage, y=residuos, size=cooks_distance,
                   sizes=(50, 200), alpha=0.6, ax=ax)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    k = 2  # Número de parâmetros (intercepto + X)
    n = len(residuos)
    ax.axvline(2 * k / n, color='red', linestyle=':', label='Limite Leverage')
    ax.set_title('Resíduos vs Leverage')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Resíduos')
    ax.legend(fontsize = 8)




'''
Resumo das Regiões Críticas
Leverage (h_i)	Resíduo	Interpretação
Alto	Alto	Ponto influente (investigar!)
Alto	Baixo	Ponto alavancado (pode ser benigno)
Baixo	Alto	Outlier
Baixo	Baixo	Observação normal

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

Exemplo Prático no Código Original
No seu código, o gráfico é gerado por:

python
sns.residplot(x=fitted, y=residuos, lowess=True, 
             scatter_kws={'alpha': 0.6}, 
             line_kws={'color': 'red', 'lw': 1})
lowess=True: Adiciona a linha de tendência suavizada.

scatter_kws: Controla a transparência dos pontos.

Resumo dos Problemas e Soluções
Padrão no Gráfico	Problema	Solução
Dispersão aleatória	Homocedasticidade (OK)	Nenhuma ação necessária.
Formato de "funil"	Heterocedasticidade	Transformar y (ex.: log(y)), WLS.
Curva ou "U"	Não-linearidade	Adicionar X², X³ ou usar GAM.
Pontos isolados extremos	Outliers	Investigar dados, usar RANSAC.
Dica Final
Combine esse gráfico com os outros do seu painel de diagnósticos:

Q-Q Plot: Verifica normalidade dos resíduos.

Scale-Location: Confirma heterocedasticidade.

Histograma: Mostra a distribuição dos resíduos.

Se precisar de ajuda para ajustar o modelo com base nos diagnósticos, posso sugerir códigos específicos!



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
'''