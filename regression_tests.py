import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
#import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

sns.set_style("whitegrid")


def test_residues(model, tests=['KS', 'Shapiro', 'Anderson','VIF'], alpha=0.05, rigor=2, plot=True, vprint=False):
    """
    Realiza múltiplos testes de normalidade nos resíduos e retorna um DataFrame consolidado.
    
    Parâmetros:
    -----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo de regressão ajustado
    tests : list, optional
        Lista dos testes a serem realizados ('KS', 'Shapiro', 'Anderson','VIF')
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
    VIF_result = None
    
    if 'KS' in tests:
        ks_result = KS_test(model, alpha=alpha)
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
        anderson_result = anderson_test(model, rigor=rigor, vprint=vprint)
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
    print(df_results)
    
    return (df_results)

def VIF_test(model):    
    # Recupera dados do modelo
    X_with_const = model.model.exog
    if X_with_const.shape[1] == 2:  # 1 variável + constante
        print('Regressão linear simples - Não será calculado o VIF')
        return

        

    X_original = pd.DataFrame(X_with_const[:, 1:])  # Pega apenas a coluna da variável (exclui constante)
    variable_names = model.params.index.tolist()    
    X_original.columns = variable_names[1:]
    # Calcula VIF para cada variável
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X_original.columns
    vif_data["VIF"] = [variance_inflation_factor(X_original.values, i) for i in range(X_original.shape[1])]
    vif_data["Tolerância"] = 1 / vif_data["VIF"]  # Tolerância = 1/VIF
    vif_data["Resultado"] = np.where(
        vif_data["Tolerância"] < 0.1,
        "Possível multicolinearidade",  # Value if condition is True
        "OK"                           # Value if condition is False
    )
    
    print(vif_data)
    return(vif_data)


def KS_test(model, alpha=0.05):
    '''
    Realiza o teste KS, comparando com a distribuição normal padrão e
        cria os gráficos correspondentes.
    Parâmetros:
        model : statsmodels.regression.linear_model.RegressionResultsWrapper
            Modelo de regressão ajustado
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


def anderson_test(model, rigor=2, vprint=False):
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

def diagnostic_plots(model, plots=['regressao','residuos', 'qq', 'hist', 'scale', 'cook','leverage','KS'], 
                     n_cols=2, figsize=None, save_path = None, return_fig=False):
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
        'regressao':(plot_regression,(model,)),
        'residuos': (plot_residuos_vs_ajustados, (fitted, residuos)),
        'qq': (plot_qq, (residuos,)),
        'hist': (plot_hist_residuos, (residuos,)),
        'scale': (plot_scale_location, (fitted, residuos)),
        'cook':  (plot_cook, (residuos, cooks_distance)),
        'leverage': (plot_leverage, (residuos, leverage, cooks_distance)),
        'KS':(plot_KS, (fitted, residuos))
    }
    
    # Filtra apenas os plots solicitados, existentes e não duplicados
    valid_plots = list(dict.fromkeys(p for p in plots if p in plot_functions))
    n_plots = len(valid_plots)
    invalid_plots = [p for p in plots if p not in plot_functions]
    all_plots = [p for p in plot_functions]
    
    if invalid_plots:
        print(f'Plots inválidos detectados e ignorados: {invalid_plots}.\n'
            f'Opções válidas: {all_plots}')
    if n_plots == 0:
        raise ValueError(f'Nenhum gráfico válido selecionado. \n'
                         f'Opções válidas: {all_plots}')
    
    # Configura layout dos subplots
    n_cols = min(n_cols, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    if figsize is None:
        base_width = 5
        base_height = 4
        figsize = (n_cols*base_width,n_rows*base_height)
    
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
    if save_path is not None:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        except:
            print('Falha ao salvar a figura. Verifique se o caminho informado é válido.')
    plt.show()
    if return_fig:
        return(fig)

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
    ax.axhline(0, color='green', linestyle='--', linewidth=0.8)
    k = 2  # Número de parâmetros (intercepto + X)
    n = len(residuos)
    ax.axvline(2 * k / n, color='red', linestyle=':', label='Limite Leverage')
    ax.set_title('Resíduos vs Leverage')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Resíduos')
    ax.legend(fontsize = 8)

def plot_cook(residuos,  cooks_distance, ax):
    ''' Gráfico com a distância de Cook'''
    ax.stem(cooks_distance,linefmt=':')
    ax.axhline(y=4/len(residuos), color='r', linestyle='--')  # Limiar
    ax.set_xlabel('Resíduos')
    ax.set_ylabel('Distância de Cook')
    ax.set_title('Distância de Cook')

def plot_regression(model, ax=None):
    """
    Plota a regressão linear com intervalo de confiança.
    Para regressão simples: mostra o gráfico de X vs Y.
    Para regressão múltipla: mostra o gráfico de valores ajustados vs observados.
    """
    if ax is None:
        ax = plt.gca()
    
    # Recuperar dados do modelo
    X_with_const = model.model.exog
    y_original = model.model.endog
    
    # Verifica se é regressão simples (apenas 1 variável independente + constante)
    if X_with_const.shape[1] == 2:  # 1 variável + constante
        X_original = X_with_const[:, 1]  # Pega apenas a coluna da variável (exclui constante)
        
        # Previsões e intervalo de confiança
        predictions = model.get_prediction(X_with_const)
        pred_df = predictions.summary_frame(alpha=0.05)
        
        # Ordena os valores para plotar a linha suave
        sorted_idx = np.argsort(X_original)
        X_sorted = X_original[sorted_idx]
        y_pred_sorted = model.fittedvalues[sorted_idx]
        ci_lower_sorted = pred_df['obs_ci_lower'][sorted_idx]
        ci_upper_sorted = pred_df['obs_ci_upper'][sorted_idx]
        
        # Plot
        ax.scatter(X_original, y_original, color='blue', alpha=0.6, label='Dados')
        ax.plot(X_sorted, y_pred_sorted, 'r-', label='Regressão')
        ax.fill_between(
            X_sorted,
            ci_lower_sorted,
            ci_upper_sorted,
            color='gray',
            alpha=0.2,
            label='IC 95%'
        )
        ax.set_xlabel('Variável Independente')
        ax.set_ylabel('Variável Dependente')
        
    else:  # Regressão múltipla
        y_pred = model.fittedvalues
        ax.scatter(y_pred, y_original, color='blue', alpha=0.6, label='Observado vs Ajustado')
        ax.plot([y_original.min(), y_original.max()], 
                [y_original.min(), y_original.max()], 
                'r--', label='Linha de Igualdade')
        ax.set_xlabel('Valores Ajustados')
        ax.set_ylabel('Valores Observados')
    
    ax.legend()
    ax.grid(True)
    ax.set_title('Gráfico da Regressão')