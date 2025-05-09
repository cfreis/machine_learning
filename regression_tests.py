import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence


sns.set_style("whitegrid")


def test_residues(model, tests=['KS', 'Shapiro', 'Anderson'], alpha=0.05, rigor=2, plot=True, vprint=False):
    """
    Realiza múltiplos testes de normalidade nos resíduos e retorna um DataFrame consolidado.
    
    Parâmetros:
    -----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo de regressão ajustado
    tests : list, optional
        Lista dos testes a serem realizados ('KS', 'Shapiro', 'Anderson')
    alpha : float, optional
        Nível de significância para KS e Shapiro-Wilk (default=0.05)
    rigor : int, optional
        Nível de rigor para Anderson-Darling (0-4) (default=2 para 5%)
    plot : bool, optional
        Se True, mostra gráficos dos testes que possuem visualização (default=True)
    vprint : bool, optional
        Se True, mostra valores críticos detalhados (default=False)
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame com colunas: ['Teste', 'Estatística', 'Valor_Referência', 'Resultado']
    """
    results = []
    
    if 'KS' in tests:
        ks_result = KS_test(model, plot=plot, alpha=alpha)
        teste, estatistica, valor_ref, resultado = ks_result.split(';')
        results.append({
            'Teste': teste,
            'Estatística': float(estatistica),
            'p-value': valor_ref,
            'Valor_Referência': f'p-valor={alpha}',
            'Resultado': resultado
        })
    
    if 'Shapiro' in tests:
        shapiro_result = shapiro_test(model, alpha=alpha)
        teste, estatistica, valor_ref, resultado = shapiro_result.split(';')
        results.append({
            'Teste': teste,
            'Estatística': float(estatistica),
            'p-value': valor_ref,
            'Valor_Referência': f'p-valor={alpha}',
            'Resultado': resultado
        })
    
    if 'Anderson' in tests:
        anderson_result = anderson_tests(model, rigor=rigor, vprint=vprint)
        teste, estatistica, valor_ref, resultado, signif = anderson_result.split(';')
        results.append({
            'Teste': teste,
            'Estatística': float(estatistica),
            'p-value': valor_ref,
            'Valor_Referência': f'Significancia={signif}',
            'Resultado': resultado
        })
    
    # Cria DataFrame
    df_results = pd.DataFrame(results)
    
    return df_results


def KS_test(model, alpha=0.05):
    '''
    Realiza o teste KS, comparando com a distribuição normal padrão e
        cria os gráficos correspondentes.
    Parâmetros:
        model : statsmodels.regression.linear_model.RegressionResultsWrapper
            Modelo de regressão ajustado
        plot : Boolean, opcional
            imprime o gráfico KS - default=True
        alpha : float, opcional
            valor para comparação com o p-valor - default=0.05
    '''
    residuos, fitted = extrai_dados(model)
    KS_statistic, KS_p_value = stats.kstest(residuos, 'norm')  # 'norm' = distribuição normal padrão
    
    result = f'Kolmogorov-Smirnov;{KS_statistic:.4f};{KS_p_value:.4f};H0=Distr Normal -> '
    
    # Interpretação
    if KS_p_value > alpha:
        result = f'{result}Não rejeitar'
    else:
        result = f'{result}Rejeitar'
        
    return(result)



def shapiro_test(model, alpha = 0.05):
    '''
    Realiza o teste de normalidade de Shapiro-Wilk
    Parâmetros:
        model : statsmodels.regression.linear_model.RegressionResultsWrapper
            Modelo de regressão ajustado
        alpha : float, opcional
            valor para comparação com o p-valor - default=0.05
    '''
    residuos, fitted = extrai_dados(model)

    # Teste para normalidade
    sh_statistic, sh_p_value = stats.shapiro(residuos)
    
    result = f'Shapiro-Wilk;{sh_statistic:.4f};{sh_p_value:.4f};H0=Distr Normal -> '
    
    # Interpretação
    if sh_p_value > alpha:
        result = f'{result}Não rejeitar'
    else:
        result = f'{result}Rejeitar'

    return(result)


def anderson_tests(model, rigor=2, vprint=False):
    '''
    Realiza o teste KS, comparando com a distribuição normal padrão e
        cria os gráficos correspondentes.
    Parâmetros:
        model : statsmodels.regression.linear_model.RegressionResultsWrapper
            Modelo de regressão ajustado
        rigor : Inteiro, opcional
            Informa o nível de significância a ser adotado
             0 - 15% (menos rigoroso)
             1 - 10%
             2 - 5% (mais usado)
             3 - 2.5%
             4 - 1 (mais rigoroso)
        print : boolean, opcional
            Imprime oa valores críticos e níveis de significância
    '''

    residuos, fitted = extrai_dados(model)
    
    # Teste para normalidade
    resultado = stats.anderson(residuos, dist='norm')

    result = f'Anderson-Darling;{resultado.statistic:.4f};{resultado.critical_values[rigor]:.4f};H0=Distr Normal -> '
    
    if vprint:    
        print("Valores críticos:", resultado.critical_values)
        print("Níveis de significância (%):", resultado.significance_level)
    
    # Interpretação
    if resultado.statistic > resultado.critical_values[rigor]:  # Compara com o valor crítico para 5%
        result = f'{result}Rejeitar'
    else:
        result = f'{result}Não rejeitar'
    result = f'{result};{resultado.significance_level[rigor]}%'
        
    return(result)

def extrai_dados(model):
    residuos = model.resid
    fitted = model.fittedvalues

    return(residuos, fitted)

def diagnostic_plots(model, plots=['residuos', 'qq', 'hist', 'scale', 'leverage','KS'], n_cols=2, figsize=(0, 0)):
    """
    Cria subplots dinâmicos com os gráficos de diagnóstico selecionados.
    
    Parâmetros:
    -----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo de regressão ajustado
    plots : list, optional
        Lista dos gráficos a incluir (opções: 'residuos', 'qq', 'hist', 'scale', 'leverage','KS')
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
        'leverage': (plot_leverage, (residuos, leverage, cooks_distance)),
        'KS':(plot_KS, (fitted, residuos))
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


def plot_KS(fitted, residuos, ax):
    # Calcular CDF empírica e teórica
    x = np.linspace(-4, 4, 1000)
    cdf_empirica = np.array([np.sum(residuos <= xx)/len(residuos) for xx in x])
    cdf_teorica = stats.norm.cdf(x)
    
    # Encontrar o ponto de máxima diferença
    idx_max = np.argmax(np.abs(cdf_empirica - cdf_teorica))
    
    KS_statistic, KS_p_value = stats.kstest(residuos, 'norm') 
    
    # Plotar
    ax.plot(x, cdf_teorica, label='CDF Teórica N(0,1)')
    ax.plot(x, cdf_empirica, label='CDF Empírica')
    ax.vlines(x=x[idx_max], ymin=cdf_teorica[idx_max], ymax=cdf_empirica[idx_max], 
               color='red', linestyle='--', 
               label=f'Diferença máxima: {KS_statistic:.3f}')
    ax.set_title('Teste de Kolmogorov-Smirnov')
    ax.set_xlabel('Valores')
    ax.set_ylabel('Probabilidade Acumulada')
    ax.legend()
    
   
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


