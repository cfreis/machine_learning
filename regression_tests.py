#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:41:36 2025

@author: Clovis F Reis


"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
#import warnings
import os
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.graphics.regressionplots import plot_partregress_grid
from statsmodels.graphics.gofplots import qqplot 
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import variance_inflation_factor

__all__ = ['test_residues',
           'diagnostic_plots',
           'VIF_test',
           'summary']

sns.set_style("whitegrid")

def _cria_modelo(df):
    y = df.iloc[:,0]
    df1 = df.iloc[:,1:]
    df1 = sm.add_constant(df1)
    model = sm.OLS(y, df1).fit()
    return(df, model)

def _prepare_data(data):
    '''prepara os dados de forma a padronizar as entradas
        Entradas possíveis são:
            Dicionário, Dataframe ou modelo OLS
        Retorna um dataframe contendo Y, os X e a constante, mais um modelo OLS
    '''
    
    if isinstance(data, pd.DataFrame):
        df, model = _cria_modelo(data)
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
        df, model = _cria_modelo(df)
    elif "statsmodels.regression.linear_model.RegressionResultsWrapper" in str(type(data)):  # Verifica se é algo do statsmodels
        df = None
        model = data
        
    return(df,model)
    
def summary(data, beta=False):
    df, model = _prepare_data(data)
    
    if beta:
        # Extrai os coeficientes não padronizados (excluindo o intercepto)
        coef_nao_padronizados = model.params.drop("const")
        
        # Calcula os desvios padrão
        std_X = df.iloc[:,1:].std(ddof=0) # DP das variáveis independentes
        std_y = df.iloc[:,0].std(ddof=0)                # DP da variável dependente

        # Coeficientes padronizados (beta)
        coef_padronizados = coef_nao_padronizados * (std_X / std_y)
        
        # Adiciona o intercepto (zero, pois y e X são centralizados)
        coef_padronizados = pd.concat([
            pd.Series({"const": 0.0}),  # Intercepto padronizado é sempre zero
            coef_padronizados
        ])
        
        
        
        # DataFrame final
        coef_df = pd.DataFrame({
            "Variável": coef_padronizados.index,
            "Coef(B)": model.params,
            "CPad(Beta)": coef_padronizados,
            "Erro_Pad": model.bse,
            "t": model.tvalues,
            "p-Val": model.pvalues,
            "IC2.5%": model.conf_int()[0],
            "IC97.5%": model.conf_int()[1]
        }).set_index("Variável")
        
        coef_df = coef_df.round(3)
        coef_df["CPad(Beta)"] = coef_df["CPad(Beta)"].astype(str)
        
        coef_df.iloc[0,1]=''
        
        print(model.summary().tables[0])
        print(coef_df)
        print(model.summary().tables[2])
        print(model.summary().extra_txt)
    
    else:
        print(model.summary())


    return(model)

def test_residues(data, tests=['KS', 'Shapiro', 'Anderson','VIF'], alpha=0.05, rigor=2, plot=True, vprint=False):
    """
    Realiza múltiplos testes de normalidade nos resíduos e retorna um DataFrame consolidado.
    
    Parâmetros:
    -----------
    data : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo de regressão ajustado;
            dicionário ou dataframe contendo os dados
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
    
    df, model = _prepare_data(data)
    results = []
    VIF_result = None
    
    if 'KS' in tests:
        ks_result = _KS_test(model, alpha=alpha)
        teste, estatistica, valor_ref, resultado = ks_result.split(';')
        results.append({
            'Teste': teste,
            'Estatística': float(estatistica),
            'p-value': valor_ref,
            'Valor_Referência': f'p-valor={alpha}',
            'Resultado': resultado
        })
    
    if 'Shapiro' in tests:
        shapiro_result = _shapiro_test(model, alpha=alpha)
        teste, estatistica, valor_ref, resultado = shapiro_result.split(';')
        results.append({
            'Teste': teste,
            'Estatística': float(estatistica),
            'p-value': valor_ref,
            'Valor_Referência': f'p-valor={alpha}',
            'Resultado': resultado
        })
    
    if 'Anderson' in tests:
        anderson_result = _anderson_test(model, rigor=rigor, vprint=vprint)
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
    
    return (model)

def VIF_test(data):    
    
    df, model = _prepare_data(data)

    # Recupera dados do modelo
    X_with_const = model.model.exog
    if X_with_const.shape[1] == 2:  # 1 variável + constante
        print('Regressão linear simples - Não será calculado o VIF')
        return(model)

        

    X_original = pd.DataFrame(X_with_const[:, 1:])  # Pega apenas a coluna da variável (exclui constante)
    variable_names = model.params.index.tolist()    
    X_original.columns = variable_names[1:]
    # Calcula VIF para cada variável
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X_original.columns
    vif_data["VIF"] = [variance_inflation_factor(X_original.values, i) for i in range(X_original.shape[1])]
    vif_data["Tolerância"] = 1 / vif_data["VIF"]  # Tolerância = 1/VIF
    vif_data["Multicolinearidade"] = np.select(
        condlist=[
            vif_data["Tolerância"] >= 0.2,                
            (vif_data["Tolerância"] >= 0.1) & (vif_data["Tolerância"] < 0.2),  
            vif_data["Tolerância"] < 0.1                
        ],
        choicelist=[
            "      Ausente     ",               
            "      Moderada    ",       
            "      Severa      "    
        ],
        default="Ocorreu um erro  - valor fora das faixas" )   
    print(vif_data)
    return(vif_data)


def _KS_test(model, alpha=0.05):
    '''
    Realiza o teste KS, comparando com a distribuição normal padrão e
        cria os gráficos correspondentes.
    Parâmetros:
        model : statsmodels.regression.linear_model.RegressionResultsWrapper
            Modelo de regressão ajustado
        alpha : float, opcional
            valor para comparação com o p-valor - default=0.05
    '''
    residuos, fitted = _extrai_dados(model)
    KS_statistic, KS_p_value = stats.kstest(residuos, 'norm')  # 'norm' = distribuição normal padrão
    
    result = f'Kolmogorov-Smirnov;{KS_statistic:.4f};{KS_p_value:.4f};H0=Distr Normal -> '
    
    # Interpretação
    if KS_p_value > alpha:
        result = f'{result}Não rejeitar'
    else:
        result = f'{result}Rejeitar'
        
    return(result)



def _shapiro_test(model, alpha = 0.05):
    '''
    Realiza o teste de normalidade de Shapiro-Wilk
    Parâmetros:
        model : statsmodels.regression.linear_model.RegressionResultsWrapper
            Modelo de regressão ajustado
        alpha : float, opcional
            valor para comparação com o p-valor - default=0.05
    '''
    residuos, fitted = _extrai_dados(model)

    # Teste para normalidade
    sh_statistic, sh_p_value = stats.shapiro(residuos)
    
    result = f'Shapiro-Wilk;{sh_statistic:.4f};{sh_p_value:.4f};H0=Distr Normal -> '
    
    # Interpretação
    if sh_p_value > alpha:
        result = f'{result}Não rejeitar'
    else:
        result = f'{result}Rejeitar'

    return(result)


def _anderson_test(model, rigor=2, vprint=False):
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

    residuos, fitted = _extrai_dados(model)
    
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

def _extrai_dados(model):
    residuos = model.resid
    fitted = model.fittedvalues

    return(residuos, fitted)

def diagnostic_plots(data, plots=['regressao','residuos', 'qq', 'hist', 'scale', 'cook','leverage','KS','multicol'], 
                     n_cols=2, figsize=None, save_path = None, return_fig=False):
    """
    Cria subplots dinâmicos com os gráficos de diagnóstico selecionados.
    
    Parâmetros:
    -----------
    data : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo de regressão ajustado;
            dicionário ou dataframe contendo os dados
    plots : list, optional
        Lista dos gráficos a incluir (opções: 'residuos', 'qq', 'hist', 'scale', 'leverage','KS','multicol')
    n_cols : inteiro, opcional
        Túmero de colunas de gráficos no sub-plot. Default=2
    figsize : tuple, optional
        Tamanho da figura (largura, altura)
    """
    df, model = _prepare_data(data)
    
    #prepara o nome das figuras
    if save_path is not None:
        path, extensao = os.path.splitext(save_path)
        if extensao == '':
            extensao = '.png'
    # Complemento dos nomes para qdo salvar mais de uma figura
    figs = ['','_reg','_mult']

    # número de variáveis independentes
    var_ind = df.shape[1] -1
    
    # Dados do modelo
    fitted = model.fittedvalues
    residuos = model.resid
    influence = OLSInfluence(model)
    leverage = influence.hat_matrix_diag
    cooks_distance = influence.cooks_distance[0]
    
    # Mapeamento das funções de plot
    plot_functions = {
        'regressao':(_plot_regression,(model,df,var_ind)),
        'residuos': (_plot_residuos_vs_ajustados, (fitted, residuos)),
        'qq': (_plot_qq, (residuos,)),
        'hist': (_plot_hist_residuos, (residuos,)),
        'scale': (_plot_scale_location, (fitted, residuos)),
        'cook':  (_plot_cook, (residuos, cooks_distance)),
        'leverage': (_plot_leverage, (residuos, leverage, cooks_distance)),
        'KS':(_plot_KS, (fitted, residuos)),
        'multicol': (_plot_multicol,(df,))
    }
    
    # Filtra apenas os plots solicitados, existentes e não duplicados
    valid_plots = list(dict.fromkeys(p for p in plots if p in plot_functions))
    # gráfico de multicolinearidade é sempre exclusivo e só é criado se 
    # var_ind > 1
    n_plots = len(valid_plots)
    multi = False
    regr = False
    
    # Adequa o tamanho do grid da figura aos graficos
    # Gráficos de multicolinearidade só será impresso para 2 ou mais var independentes
    # O gráfico de regressão é impresso junto com os outros se há somente uma var independ
    # e separado se há 2 ou mais
    if ('multicol' in valid_plots):
        valid_plots.remove('multicol')
        if var_ind > 1:
            multi = True
        else:
            print('Gráfico de multicolinearidade não será exibido, pois há somente uma variável independente')
  
    if ('regressao' in valid_plots) and (var_ind > 1):
        valid_plots.remove('regressao')
        regr = True

    grid_plots = len(valid_plots)

    invalid_plots = [p for p in plots if p not in plot_functions]
    all_plots = [p for p in plot_functions]
    
    if invalid_plots:
        print(f'Plots inválidos detectados e ignorados: {invalid_plots}.\n'
            f'Opções válidas: {all_plots}')
    if n_plots == 0:
        raise ValueError(f'Nenhum gráfico válido selecionado. \n'
                         f'Opções válidas: {all_plots}')
    
    # Configura layout dos subplots
    try:
        n_cols_calc = min(n_cols, grid_plots)
        n_rows = int(np.ceil(grid_plots / n_cols_calc))
    except ZeroDivisionError:
        n_cols_calc = 1
        n_rows = 1
    if figsize is None:
        base_width = 5
        base_height = 4
        figsize = (n_cols_calc*base_width,n_rows*base_height)
    
    fig, axes = plt.subplots(n_rows, n_cols_calc, figsize=figsize)
    if grid_plots == 1:
        axes = np.array([axes])  # Garante que axes seja sempre um array
        
    # Achata o array de axes para facilitar iteração
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Gera cada gráfico
    idx_ax = 0
    # for i, plot_name in enumerate(valid_plots):
    while idx_ax < grid_plots:
        plot_name = valid_plots[idx_ax]
        if (plot_name == 'regressao') and (regr):
            continue

        ax = axes_flat[idx_ax]
        plot_func, args = plot_functions[plot_name]
        plot_func(*args, ax=ax)
        idx_ax = idx_ax + 1

   # Remove eixos vazios
    for j in range(idx_ax, len(axes_flat)):
        axes_flat[j].axis('off')
    
    if regr:
        fig_reg = plot_partregress_grid(model, fig=plt.figure(figsize=(12, 8)))
        fig_reg.suptitle("Efeitos Parciais das Variáveis Independentes", fontsize=16,y=1.05)

    
    if multi:
        fig_multi = sns.pairplot(df.iloc[:,1:], kind='reg',diag_kind='hist',plot_kws={'line_kws':{'color':'red'}})
        fig_multi.fig.suptitle("Análise de Multicolinearidade", fontsize=16,y=1.05)
        #fig_multi.fig.subplots_adjust(top=0.1) 
    if save_path is not None:
        try:
            fig_name = f'{path}{figs[0]}{extensao}'
            fig.savefig(fig_name, bbox_inches='tight', dpi=300)
            if regr:
                fig_name = f'{path}{figs[1]}{extensao}'
                fig_reg.savefig(fig_name, bbox_inches='tight', dpi=300)
            if multi:
                fig_name = f'{path}{figs[2]}{extensao}'
                fig_multi.savefig(fig_name, bbox_inches='tight', dpi=300)
        except:
            print('Falha ao salvar a figura. Verifique se o caminho informado é válido.')

    #     # gráfico de multicolinearidade somente se há mais de 1 var independente
    #     if plot_name == 'multicol':
    #         if var_ind > 1:
    #             fig_mult, axes_mult = plt.subplots(var_ind, var_ind, figsize=figsize)
    #             plot_func, args = plot_functions[plot_name]
    #             plot_func(*args, ax=axes_reg)
 

    plt.tight_layout()    
    plt.show()
    if return_fig:
        return(fig, model)
    return(model)

def _plot_KS(fitted, residuos, ax):
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
    
   
def _plot_residuos_vs_ajustados(fitted, residuos, ax):
    """Gráfico de Resíduos vs Valores Ajustados"""
    sns.residplot(x=fitted, y=residuos, lowess=True,
                 scatter_kws={'alpha': 0.6},
                 line_kws={'color': 'red', 'lw': 1}, ax=ax)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title('Resíduos vs Ajustados')
    ax.set_xlabel('Valores Ajustados')
    ax.set_ylabel('Resíduos')

def _plot_qq(residuos, ax):
    """Gráfico Q-Q para normalidade"""
    qqplot(residuos, line='s', ax=ax)
    ax.set_title('Q-Q Plot')
    ax.get_lines()[0].set_markersize(4.0)  # Ajusta tamanho dos pontos


def _plot_hist_residuos(residuos, ax):
    """Histograma dos Resíduos"""
    sns.histplot(residuos, kde=True, ax=ax)
    ax.set_title('Distribuição dos Resíduos')
    ax.set_xlabel('Resíduos')
   
def _plot_scale_location(fitted, residuos, ax):
    """Gráfico Scale-Location"""
    residuos_norm = np.sqrt(np.abs(residuos / residuos.std()))
    sns.regplot(x=fitted, y=residuos_norm, lowess=True,
               scatter_kws={'alpha': 0.6},
               line_kws={'color': 'red', 'lw': 1}, ax=ax)
    ax.set_title('Scale-Location')
    ax.set_xlabel('Valores Ajustados')
    ax.set_ylabel('√(|Resíduos Padronizados|)')

def _plot_leverage(residuos, leverage, cooks_distance, ax):
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

def _plot_cook(residuos,  cooks_distance, ax):
    ''' Gráfico com a distância de Cook'''
    ax.stem(cooks_distance,linefmt=':')
    ax.axhline(y=4/len(residuos), color='r', linestyle='--')  # Limiar
    ax.set_xlabel('Resíduos')
    ax.set_ylabel('Distância de Cook')
    ax.set_title('Distância de Cook')

def _plot_regression_loop(model,x, x_name, y, y_name, ax):
    
    # Previsões e intervalo de confiança
    Xwith_const = model.model.exog
    predictions = model.get_prediction(Xwith_const)
    pred_df = predictions.summary_frame(alpha=0.05)
    
    # Ordena os valores para plotar a linha suave
    sorted_idx = np.argsort(x)
    x = x[sorted_idx]
    y_pred_sorted = model.fittedvalues[sorted_idx]
    ci_lower_sorted = pred_df['obs_ci_lower'][sorted_idx]
    ci_upper_sorted = pred_df['obs_ci_upper'][sorted_idx]
    
    # Plot
    ax.scatter(x, y, color='blue', alpha=0.6, label='Dados')
    ax.plot(x, y_pred_sorted, 'r-', label='Regressão')
    ax.fill_between(
        x,
        ci_lower_sorted,
        ci_upper_sorted,
        color='gray',
        alpha=0.2,
        label='IC 95%'
    )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

# def _plot_regr_multi(model,df, var_ind,n_cols,figsize,ax):
#     n_cols_reg = min(n_cols, var_ind)
#     n_rows_reg = int(np.ceil(var_ind / n_cols_reg))
#     fig_reg, axes_reg = plt.subplots(n_rows_reg,n_cols_reg, figsize=figsize)
#     # plot_func, args = plot_functions[plot_name]
#     # plot_func(*args, ax=axes_reg)

#     X = df.iloc[:,1:]
#     X_names = df.columns[1:]
#     y = df.iloc[:,0]
#     y_name = df.columns[0]
    
#     fig, axes = plt.subplots(n_cols_reg, n_rows_reg, figsize=figsize)
        
#     # Achata o array de axes para facilitar iteração
#     axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
#     # Gera cada gráfico
#     idx_ax = 0
#     # for i, plot_name in enumerate(valid_plots):
#     while idx_ax < var_ind:
#         x = X[:,idx_ax]
#         x_name = X_names[idx_ax]
#         ax = axes_flat[idx_ax]
#         _plot_regression_loop(model,x, x_name, y, y_name, ax)
#         idx_ax = idx_ax + 1

#    # Remove eixos vazios
#     for j in range(idx_ax, len(axes_flat)):
#         axes_flat[j].axis('off')

#     plt.show()

def _plot_regression(model,df, var_ind, ax):
    """
    Plota a regressão linear com intervalo de confiança.
    Para regressão simples: mostra o gráfico de X vs Y.
    Para regressão múltipla: mostra o gráfico de valores ajustados vs observados.
    """

    # Recuperar dados do modelo
    # Xwith_const = model.model.exog
    # y_original = model.model.endog
    
    if var_ind < 1:
        raise ValueError('Há alguma inconsistência nos dados. Por favor verifique.')
        
    
    # Verifica se é regressão simples (apenas 1 variável independente + constante)
    if var_ind == 1:  # 1 variável independente
        x = df.iloc[:,1]
        x_name = df.columns[1]
        y = df.iloc[:,0]
        y_name = df.columns[0]
        _plot_regression_loop(model,x, x_name, y, y_name, ax)
    
    ax.legend()
    ax.grid(True)
    ax.set_title('Gráfico da Regressão')
    
def _plot_multicol(df):
    return 0