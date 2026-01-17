"""
Laboratório de Econometria - Module 10: Panel Data
Aplicativo educacional interativo para dados em painel, efeitos fixos/aleatórios e aplicações.
Público-alvo: alunos de MBA com perfis quantitativos heterogêneos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# =============================================================================
# FUNÇÕES AUXILIARES PARA SIMULAÇÃO E CÁLCULOS
# =============================================================================

def make_panel_data(n_entities: int = 10, n_periods: int = 20, beta: float = 0.5,
                    fe_variance: float = 2.0, error_variance: float = 1.0,
                    seed: int = 42) -> pd.DataFrame:
    """
    Gera painel balanceado com efeitos fixos.
    y_it = alpha_i + beta * x_it + epsilon_it
    """
    np.random.seed(seed)
    
    # Efeitos fixos por entidade
    alpha = np.random.normal(0, np.sqrt(fe_variance), n_entities)
    
    data = []
    for i in range(n_entities):
        for t in range(n_periods):
            x = np.random.normal(5, 2)
            epsilon = np.random.normal(0, np.sqrt(error_variance))
            y = alpha[i] + beta * x + epsilon
            
            data.append({
                'entity': f'Banco_{i+1}',
                'entity_id': i,
                'period': t + 1,
                'x': x,
                'y': y,
                'alpha_true': alpha[i]
            })
    
    return pd.DataFrame(data)


def make_unbalanced_panel(n_entities: int = 10, n_periods: int = 20, 
                          missing_prob: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Gera painel não balanceado com observações faltantes."""
    np.random.seed(seed)
    
    df = make_panel_data(n_entities, n_periods, seed=seed)
    
    # Remover observações aleatoriamente
    mask = np.random.random(len(df)) > missing_prob
    df_unbalanced = df[mask].copy()
    
    return df_unbalanced


def make_sur_data(n_obs: int = 100, rho: float = 0.7, seed: int = 42) -> dict:
    """
    Gera dados para SUR (Seemingly Unrelated Regressions).
    Duas equações com erros correlacionados.
    """
    np.random.seed(seed)
    
    # Regressores
    x1 = np.random.normal(0, 1, n_obs)
    x2 = np.random.normal(0, 1, n_obs)
    
    # Erros correlacionados
    e1 = np.random.normal(0, 1, n_obs)
    e2 = rho * e1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_obs)
    
    # Equações
    y1 = 2 + 1.5 * x1 + e1  # Equação 1
    y2 = 1 + 0.8 * x2 + e2  # Equação 2
    
    return {
        'y1': y1, 'y2': y2,
        'x1': x1, 'x2': x2,
        'e1': e1, 'e2': e2,
        'rho_true': rho
    }


def fit_ols_simple(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS simples."""
    n = len(y)
    k = X.shape[1]
    
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    
    y_hat = X @ beta
    residuals = y - y_hat
    
    sse = np.sum(residuals**2)
    sst = np.sum((y - np.mean(y))**2)
    r_squared = 1 - sse / sst
    
    s2 = sse / (n - k)
    se = np.sqrt(s2 * np.diag(XtX_inv))
    
    return {
        'beta': beta,
        'se': se,
        'r_squared': r_squared,
        'residuals': residuals,
        'y_hat': y_hat,
        's2': s2
    }


def fit_pooled_ols(df: pd.DataFrame) -> dict:
    """Pooled OLS (ignora estrutura de painel)."""
    y = df['y'].values
    X = np.column_stack([np.ones(len(df)), df['x'].values])
    
    result = fit_ols_simple(y, X)
    
    return {
        'intercept': result['beta'][0],
        'beta': result['beta'][1],
        'se_beta': result['se'][1],
        'r_squared': result['r_squared'],
        'residuals': result['residuals']
    }


def fit_fe_simple(df: pd.DataFrame) -> dict:
    """
    Efeitos Fixos via transformação Within.
    Remove a média de cada entidade.
    """
    # Calcular médias por entidade
    entity_means = df.groupby('entity_id')[['y', 'x']].transform('mean')
    
    # Transformação within
    y_within = df['y'].values - entity_means['y'].values
    x_within = df['x'].values - entity_means['x'].values
    
    # OLS sem intercepto
    X = x_within.reshape(-1, 1)
    y = y_within
    
    beta = np.sum(x_within * y_within) / np.sum(x_within**2)
    
    # Resíduos e R²
    y_hat = beta * x_within
    residuals = y_within - y_hat
    
    sse = np.sum(residuals**2)
    sst = np.sum(y_within**2)
    r_squared_within = 1 - sse / sst if sst > 0 else 0
    
    # Erro padrão
    n = len(df)
    n_entities = df['entity_id'].nunique()
    k = 1
    dof = n - n_entities - k
    s2 = sse / dof if dof > 0 else sse / n
    se_beta = np.sqrt(s2 / np.sum(x_within**2))
    
    # Estimar efeitos fixos
    y_means = df.groupby('entity_id')['y'].mean()
    x_means = df.groupby('entity_id')['x'].mean()
    alphas = y_means - beta * x_means
    
    return {
        'beta': beta,
        'se_beta': se_beta,
        'r_squared_within': r_squared_within,
        'alphas': alphas.values,
        'residuals': residuals,
        'n_entities': n_entities
    }


def fit_re_simple(df: pd.DataFrame) -> dict:
    """
    Efeitos Aleatórios via GLS simplificado.
    Usa transformação parcial baseada em theta.
    """
    n = len(df)
    n_entities = df['entity_id'].nunique()
    T_avg = n / n_entities
    
    # Primeiro estimar FE para obter sigma²_epsilon
    fe = fit_fe_simple(df)
    sigma2_epsilon = np.var(fe['residuals'])
    
    # Estimar sigma²_alpha (entre entidades)
    entity_means = df.groupby('entity_id')['y'].mean()
    sigma2_between = np.var(entity_means)
    sigma2_alpha = max(sigma2_between - sigma2_epsilon / T_avg, 0.001)
    
    # Theta para transformação
    theta = 1 - np.sqrt(sigma2_epsilon / (sigma2_epsilon + T_avg * sigma2_alpha))
    
    # Transformação parcial
    entity_means_y = df.groupby('entity_id')['y'].transform('mean')
    entity_means_x = df.groupby('entity_id')['x'].transform('mean')
    
    y_re = df['y'].values - theta * entity_means_y.values
    x_re = df['x'].values - theta * entity_means_x.values
    
    # OLS na variável transformada
    X = np.column_stack([np.ones(n) * (1 - theta), x_re])
    result = fit_ols_simple(y_re, X)
    
    return {
        'intercept': result['beta'][0],
        'beta': result['beta'][1],
        'se_beta': result['se'][1],
        'r_squared': result['r_squared'],
        'theta': theta,
        'sigma2_alpha': sigma2_alpha,
        'sigma2_epsilon': sigma2_epsilon,
        'residuals': result['residuals']
    }


def hausman_test_simple(df: pd.DataFrame) -> dict:
    """
    Teste de Hausman: FE vs RE.
    H0: RE é consistente e eficiente (preferível)
    H1: RE é inconsistente (usar FE)
    """
    fe = fit_fe_simple(df)
    re = fit_re_simple(df)
    
    # Diferença entre coeficientes
    diff = fe['beta'] - re['beta']
    
    # Variância da diferença (simplificado)
    var_diff = fe['se_beta']**2 - re['se_beta']**2
    var_diff = max(var_diff, 0.0001)  # Garantir positivo
    
    # Estatística de Hausman
    H = diff**2 / var_diff
    
    # P-valor (chi-quadrado com 1 gl)
    p_value = 1 - stats.chi2.cdf(H, 1)
    
    return {
        'H_stat': H,
        'p_value': p_value,
        'beta_fe': fe['beta'],
        'beta_re': re['beta'],
        'diff': diff,
        'recommendation': 'FE' if p_value < 0.05 else 'RE'
    }


def fit_sur_simple(data: dict) -> dict:
    """
    SUR simplificado: estima equações considerando correlação de erros.
    Na prática, usa OLS em cada equação e reporta correlação residual.
    """
    # OLS em cada equação
    X1 = np.column_stack([np.ones(len(data['y1'])), data['x1']])
    X2 = np.column_stack([np.ones(len(data['y2'])), data['x2']])
    
    ols1 = fit_ols_simple(data['y1'], X1)
    ols2 = fit_ols_simple(data['y2'], X2)
    
    # Correlação entre resíduos
    rho_estimated = np.corrcoef(ols1['residuals'], ols2['residuals'])[0, 1]
    
    return {
        'beta1': ols1['beta'],
        'beta2': ols2['beta'],
        'se1': ols1['se'],
        'se2': ols2['se'],
        'r2_eq1': ols1['r_squared'],
        'r2_eq2': ols2['r_squared'],
        'rho_residuals': rho_estimated,
        'residuals1': ols1['residuals'],
        'residuals2': ols2['residuals']
    }


def panel_unit_root_test_simple(df: pd.DataFrame, variable: str = 'y') -> dict:
    """
    Teste de raiz unitária em painel simplificado (tipo LLC).
    Testa se a variável é estacionária no painel.
    """
    entities = df['entity_id'].unique()
    adf_stats = []
    
    for entity in entities:
        entity_data = df[df['entity_id'] == entity][variable].values
        if len(entity_data) > 10:
            # ADF simplificado para cada entidade
            dy = np.diff(entity_data)
            y_lag = entity_data[:-1]
            
            if len(dy) > 2:
                X = np.column_stack([np.ones(len(dy)), y_lag])
                try:
                    result = fit_ols_simple(dy, X)
                    t_stat = result['beta'][1] / result['se'][1]
                    adf_stats.append(t_stat)
                except:
                    pass
    
    if len(adf_stats) > 0:
        # Média das estatísticas (LLC-type)
        avg_stat = np.mean(adf_stats)
        # P-valor aproximado
        p_value = stats.norm.cdf(avg_stat)
    else:
        avg_stat = 0
        p_value = 0.5
    
    return {
        'avg_stat': avg_stat,
        'p_value': p_value,
        'n_entities': len(adf_stats),
        'individual_stats': adf_stats
    }


def panel_cointegration_test_simple(df: pd.DataFrame) -> dict:
    """
    Teste de cointegração em painel simplificado.
    Testa se y e x são cointegrados no painel.
    """
    # Estimar relação de longo prazo por entidade
    entities = df['entity_id'].unique()
    residuals_all = []
    
    for entity in entities:
        entity_data = df[df['entity_id'] == entity]
        if len(entity_data) > 5:
            X = np.column_stack([np.ones(len(entity_data)), entity_data['x'].values])
            result = fit_ols_simple(entity_data['y'].values, X)
            residuals_all.extend(result['residuals'])
    
    residuals_all = np.array(residuals_all)
    
    # Testar estacionaridade dos resíduos
    if len(residuals_all) > 20:
        dy = np.diff(residuals_all)
        y_lag = residuals_all[:-1]
        X = np.column_stack([np.ones(len(dy)), y_lag])
        result = fit_ols_simple(dy, X)
        t_stat = result['beta'][1] / result['se'][1]
        
        # Valores críticos aproximados para painel
        critical_5 = -1.95
        p_value = stats.norm.cdf(t_stat)
    else:
        t_stat = 0
        p_value = 0.5
        critical_5 = -1.95
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'critical_5': critical_5,
        'cointegrated': t_stat < critical_5
    }


def make_banking_case_data(n_banks: int = 15, n_years: int = 10, seed: int = 42) -> pd.DataFrame:
    """Gera dados sintéticos de bancos para estudo de caso."""
    np.random.seed(seed)
    
    data = []
    
    for i in range(n_banks):
        # Características fixas do banco
        size_effect = np.random.normal(0, 1)
        efficiency = np.random.uniform(0.7, 1.0)
        
        for t in range(n_years):
            # Variáveis que mudam no tempo
            market_share = np.random.uniform(0.02, 0.15) + 0.001 * t
            cost_income = np.random.uniform(0.4, 0.7) - 0.005 * efficiency
            credit_growth = np.random.normal(0.08, 0.05)
            npl_ratio = np.random.uniform(0.02, 0.08)
            
            # ROE como variável dependente
            roe = (0.05 + 0.1 * market_share - 0.15 * cost_income 
                   + 0.02 * credit_growth - 0.3 * npl_ratio 
                   + 0.02 * size_effect + np.random.normal(0, 0.02))
            
            data.append({
                'banco': f'Banco_{chr(65+i)}',
                'banco_id': i,
                'ano': 2014 + t,
                'roe': roe * 100,  # Em %
                'market_share': market_share * 100,
                'cost_income': cost_income * 100,
                'credit_growth': credit_growth * 100,
                'npl_ratio': npl_ratio * 100,
                'size_effect': size_effect
            })
    
    return pd.DataFrame(data)


def make_credit_case_data(n_countries: int = 20, n_years: int = 15, seed: int = 42) -> pd.DataFrame:
    """Gera dados sintéticos de crédito e crescimento para estudo de caso."""
    np.random.seed(seed)
    
    data = []
    
    for i in range(n_countries):
        # Efeito fixo do país
        country_effect = np.random.normal(0, 0.5)
        base_gdp = np.random.uniform(8, 12)  # Log do PIB inicial
        
        gdp = base_gdp
        credit = base_gdp - 1  # Crédito como % do PIB (em log)
        
        for t in range(n_years):
            # Crescimento
            gdp_growth = 0.02 + 0.01 * country_effect + np.random.normal(0, 0.02)
            credit_growth = gdp_growth + 0.005 + np.random.normal(0, 0.03)
            
            # Crise em alguns anos
            if t in [5, 6] and i < n_countries // 2:
                gdp_growth -= 0.03
                credit_growth -= 0.05
            
            gdp += gdp_growth
            credit += credit_growth
            
            data.append({
                'pais': f'País_{i+1}',
                'pais_id': i,
                'ano': 2005 + t,
                'log_gdp': gdp,
                'log_credit': credit,
                'gdp_growth': gdp_growth * 100,
                'credit_gdp_ratio': np.exp(credit - gdp) * 100,
                'country_effect': country_effect
            })
    
    return pd.DataFrame(data)


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: Introdução aos Dados em Painel"""
    st.header("📊 Introdução aos Dados em Painel")
    
    st.markdown("""
    **Dados em painel** combinam duas dimensões:
    - **Cross-section:** Múltiplas unidades (empresas, países, pessoas)
    - **Séries temporais:** Observações ao longo do tempo
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("O que é um Painel?")
        
        st.markdown("""
        **Definição:**
        > Observações repetidas das **mesmas entidades** ao longo do **tempo**.
        
        **Exemplos:**
        - 50 bancos observados por 10 anos
        - 100 países medidos trimestralmente por 20 anos
        - 1000 funcionários avaliados mensalmente por 5 anos
        
        **Notação:**
        - i = entidade (1, 2, ..., N)
        - t = tempo (1, 2, ..., T)
        - y_it = valor de y para entidade i no tempo t
        """)
    
    with col2:
        st.subheader("Comparação Visual")
        
        # Criar dados para visualização
        fig = make_subplots(rows=1, cols=3, 
                           subplot_titles=["Cross-Section", "Série Temporal", "Painel"])
        
        # Cross-section: várias entidades, um período
        np.random.seed(42)
        entities_cs = [f'E{i}' for i in range(1, 8)]
        values_cs = np.random.normal(10, 2, 7)
        fig.add_trace(go.Bar(x=entities_cs, y=values_cs, marker_color='steelblue'), 
                     row=1, col=1)
        
        # Série temporal: uma entidade, vários períodos
        time_ts = list(range(1, 8))
        values_ts = np.cumsum(np.random.normal(0.5, 1, 7)) + 10
        fig.add_trace(go.Scatter(x=time_ts, y=values_ts, mode='lines+markers',
                                line=dict(color='steelblue')), row=1, col=2)
        
        # Painel: várias entidades, vários períodos
        for i in range(3):
            values_panel = np.cumsum(np.random.normal(0.3, 0.8, 7)) + 8 + i*2
            fig.add_trace(go.Scatter(x=time_ts, y=values_panel, mode='lines+markers',
                                    name=f'Entidade {i+1}'), row=1, col=3)
        
        fig.update_layout(height=300, showlegend=False)
        fig.update_xaxes(title_text="Entidade", row=1, col=1)
        fig.update_xaxes(title_text="Tempo", row=1, col=2)
        fig.update_xaxes(title_text="Tempo", row=1, col=3)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Quando Painéis Resolvem Problemas?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Heterogeneidade não observada**
        
        Cada banco tem características próprias (cultura, gestão) difíceis de medir.
        
        Painel: Controla via efeitos fixos.
        """)
    
    with col2:
        st.markdown("""
        **📈 Dinâmica temporal**
        
        Decisões de hoje afetam resultados de amanhã.
        
        Painel: Captura ajustamentos ao longo do tempo.
        """)
    
    with col3:
        st.markdown("""
        **💪 Mais poder estatístico**
        
        Cross-section: N observações
        Série: T observações
        
        Painel: N × T observações!
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa painéis quando quer comparar unidades ao longo do tempo
    - Controla para características fixas de cada unidade
    - Aumenta confiança nas conclusões causais
    """)


def render_section_S2():
    """S2: Vantagens Gerenciais dos Painéis"""
    st.header("💪 Vantagens dos Dados em Painel")
    
    st.markdown("""
    Por que painéis são tão poderosos para análise gerencial?
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Vantagem 1: Mais Dados")
        
        n_entities = st.slider("Número de bancos", 5, 30, 15, key="n_ent_v")
        n_periods = st.slider("Número de anos", 3, 20, 10, key="n_per_v")
        
        st.markdown(f"""
        **Cross-section:** {n_entities} observações
        
        **Série temporal:** {n_periods} observações
        
        **Painel:** {n_entities} × {n_periods} = **{n_entities * n_periods}** observações
        
        Mais dados = menor variância = estimativas mais precisas!
        """)
    
    with col2:
        # Visualizar aumento de dados
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Cross-Section', 'Série Temporal', 'Painel'],
            y=[n_entities, n_periods, n_entities * n_periods],
            marker_color=['steelblue', 'orange', 'green']
        ))
        
        fig.update_layout(
            title="Número de Observações",
            yaxis_title="N",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True, key=f"vant_{n_entities}_{n_periods}")
    
    st.subheader("Vantagem 2: Controle de Variável Omitida")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **O problema clássico:**
        
        Queremos estimar: ROE = β₀ + β₁ × Crédito + ε
        
        Mas "qualidade da gestão" afeta tanto Crédito quanto ROE!
        
        **Cross-section:** Viés de variável omitida
        
        **Painel com efeitos fixos:** Cada banco tem seu intercepto próprio (α_i), 
        que absorve a qualidade de gestão (constante no tempo).
        """)
    
    with col2:
        # Simular viés
        np.random.seed(42)
        n = 50
        
        # Qualidade de gestão (não observada)
        quality = np.random.normal(0, 1, n)
        
        # Crédito correlacionado com qualidade
        credit = 5 + 2 * quality + np.random.normal(0, 1, n)
        
        # ROE depende de crédito E qualidade
        roe = 10 + 0.5 * credit + 3 * quality + np.random.normal(0, 1, n)
        
        # OLS viesado
        X = np.column_stack([np.ones(n), credit])
        result = fit_ols_simple(roe, X)
        beta_biased = result['beta'][1]
        
        st.metric("β verdadeiro (Crédito → ROE)", "0.50")
        st.metric("β OLS (viés de var. omitida)", f"{beta_biased:.2f}",
                 delta=f"Viés: {beta_biased - 0.5:.2f}")
        
        st.warning("⚠️ OLS superestima o efeito porque captura parte do efeito de 'qualidade'!")
    
    with st.expander("📖 Como FE resolve o viés?"):
        st.markdown("""
        **Efeitos Fixos eliminam variação entre entidades:**
        
        Transformação Within:
        $$y_{it} - \\bar{y}_i = \\beta (x_{it} - \\bar{x}_i) + (\\varepsilon_{it} - \\bar{\\varepsilon}_i)$$
        
        - O efeito fixo α_i (incluindo qualidade de gestão) **some na transformação**
        - Usamos apenas variação **dentro** de cada entidade ao longo do tempo
        - Se qualidade é constante, não confunde mais o efeito de crédito
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa painéis para isolar efeitos de variáveis de interesse
    - Reconhece que cross-section pode ter viés de seleção
    - Prefere análises que controlem por características fixas
    """)


def render_section_S3():
    """S3: Estrutura dos Dados: Balanceado vs Não Balanceado"""
    st.header("📋 Estrutura: Balanceado vs Não Balanceado")
    
    st.markdown("""
    Na prática, painéis raramente são perfeitamente completos.
    """)
    
    tab1, tab2 = st.tabs(["📊 Painel Balanceado", "🔲 Painel Não Balanceado"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Painel Balanceado")
            
            st.markdown("""
            **Definição:** Todas as entidades observadas em todos os períodos.
            
            **Estrutura:**
            - N entidades × T períodos = N×T observações
            - Sem dados faltantes
            - Estimação mais simples
            
            **Quando ocorre:**
            - Pesquisas controladas
            - Dados administrativos completos
            - Séries financeiras padronizadas
            """)
            
            n_ent = st.slider("Entidades", 3, 10, 5, key="n_ent_bal")
            n_per = st.slider("Períodos", 3, 8, 4, key="n_per_bal")
        
        with col2:
            # Gerar painel balanceado
            df_bal = make_panel_data(n_entities=n_ent, n_periods=n_per, seed=42)
            
            # Criar matriz visual
            pivot = df_bal.pivot(index='entity', columns='period', values='y')
            
            fig = px.imshow(pivot.notna().astype(int), 
                           labels=dict(x="Período", y="Entidade", color="Observado"),
                           color_continuous_scale=['white', 'steelblue'],
                           aspect='auto')
            fig.update_layout(title="Matriz de Observações (Balanceado)", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Total de observações", f"{len(df_bal)}")
            st.metric("Completude", "100%")
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Painel Não Balanceado")
            
            st.markdown("""
            **Definição:** Algumas observações faltantes.
            
            **Causas comuns:**
            - Entrada/saída de entidades (fusões, falências)
            - Dados não reportados
            - Mudanças de amostragem
            
            **Implicações:**
            - Estimação ainda possível (com ajustes)
            - Verificar se missings são aleatórios
            - Pode indicar viés de sobrevivência
            """)
            
            missing_prob = st.slider("Prob. de missing", 0.0, 0.5, 0.2, 0.05, key="miss_prob")
        
        with col2:
            # Gerar painel não balanceado
            df_unbal = make_unbalanced_panel(n_entities=n_ent, n_periods=n_per, 
                                            missing_prob=missing_prob, seed=42)
            
            # Recriar pivô
            df_full = make_panel_data(n_entities=n_ent, n_periods=n_per, seed=42)
            df_full['observed'] = 0
            
            for _, row in df_unbal.iterrows():
                mask = (df_full['entity'] == row['entity']) & (df_full['period'] == row['period'])
                df_full.loc[mask, 'observed'] = 1
            
            pivot_unbal = df_full.pivot(index='entity', columns='period', values='observed')
            
            fig = px.imshow(pivot_unbal, 
                           labels=dict(x="Período", y="Entidade", color="Observado"),
                           color_continuous_scale=['white', 'steelblue'],
                           aspect='auto')
            fig.update_layout(title="Matriz de Observações (Não Balanceado)", height=300)
            st.plotly_chart(fig, use_container_width=True, key=f"unbal_{missing_prob}")
            
            completude = len(df_unbal) / len(df_full) * 100
            st.metric("Total de observações", f"{len(df_unbal)}")
            st.metric("Completude", f"{completude:.1f}%")
    
    with st.expander("⚠️ Cuidado: Viés de Sobrevivência"):
        st.markdown("""
        **Se missings não são aleatórios:**
        
        - Bancos que faliram saem da amostra → superestima rentabilidade
        - Empresas que param de reportar → amostra enviesada
        - Países em crise com dados atrasados → subestima efeito de crises
        
        **Soluções:**
        - Verificar padrão de missings
        - Testar se missings são aleatórios (MCAR, MAR, MNAR)
        - Usar métodos robustos (imputação, Heckman)
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Sempre verifica completude do painel
    - Investiga motivos de observações faltantes
    - Considera viés de sobrevivência nas conclusões
    """)


def render_section_S4():
    """S4: Regressões SUR: Erros que Conversam"""
    st.header("🔗 SUR: Seemingly Unrelated Regressions")
    
    st.markdown("""
    **SUR** modela sistemas de equações onde os erros são correlacionados entre equações.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Intuição")
        
        st.markdown("""
        **Cenário:** Dois bancos concorrentes no mesmo mercado.
        
        - Equação 1: Fluxo_A = α₁ + β₁×Taxa_A + ε₁
        - Equação 2: Fluxo_B = α₂ + β₂×Taxa_B + ε₂
        
        **O que conecta as equações?**
        
        Um choque no mercado (crise, regulação) afeta **ambos** os bancos simultaneamente!
        
        → Corr(ε₁, ε₂) ≠ 0
        
        **SUR aproveita essa informação:**
        - Estima equações conjuntamente
        - Ganha eficiência (erros padrão menores)
        """)
        
        rho = st.slider("Correlação entre erros (ρ)", 0.0, 0.95, 0.7, 0.05, key="rho_sur")
    
    with col2:
        # Gerar dados SUR
        data = make_sur_data(n_obs=100, rho=rho, seed=42)
        
        # Estimar
        sur = fit_sur_simple(data)
        
        st.markdown("**Resultado SUR:**")
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("ρ verdadeiro", f"{data['rho_true']:.2f}")
        col_m2.metric("ρ estimado (resíduos)", f"{sur['rho_residuals']:.2f}")
        
        # Scatter de resíduos
        fig = px.scatter(x=sur['residuals1'], y=sur['residuals2'], opacity=0.5,
                        labels={'x': 'Resíduos Eq. 1', 'y': 'Resíduos Eq. 2'})
        
        # Linha de tendência
        z = np.polyfit(sur['residuals1'], sur['residuals2'], 1)
        x_line = np.linspace(sur['residuals1'].min(), sur['residuals1'].max(), 50)
        fig.add_trace(go.Scatter(x=x_line, y=z[0]*x_line + z[1], mode='lines',
                                line=dict(color='red', dash='dash'), name='Tendência'))
        
        fig.update_layout(title="Correlação entre Resíduos", height=350)
        st.plotly_chart(fig, use_container_width=True, key=f"sur_{rho}")
    
    with st.expander("📖 Quando SUR agrega valor?"):
        st.markdown("""
        **SUR é útil quando:**
        1. Equações têm erros correlacionados
        2. Regressores são diferentes entre equações
        
        **Se regressores são iguais:** SUR = OLS equação por equação
        
        **Ganho de eficiência:**
        - Maior quando ρ é alto e regressores são diferentes
        - Menor quando ρ é baixo ou regressores são iguais
        
        **Aplicações em finanças:**
        - Retornos de ativos do mesmo setor
        - Bancos no mesmo mercado
        - Subsidiárias do mesmo grupo
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Reconhece quando choques afetam múltiplas unidades
    - Usa SUR para estimação mais eficiente
    - Interpreta correlação de erros como exposição comum
    """)


def render_section_S5():
    """S5: Efeitos Fixos vs Aleatórios"""
    st.header("⚖️ Efeitos Fixos vs Aleatórios")
    
    st.markdown("""
    Os dois principais modelos para painéis diferem em como tratam a heterogeneidade entre entidades.
    """)
    
    tab1, tab2 = st.tabs(["🔒 Efeitos Fixos (FE)", "🎲 Efeitos Aleatórios (RE)"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Efeitos Fixos (FE)")
            
            st.markdown("""
            **Intuição:**
            > "Cada entidade tem sua própria linha de base"
            
            **Modelo:**
            $$y_{it} = \\alpha_i + \\beta x_{it} + \\varepsilon_{it}$$
            
            - α_i = intercepto específico da entidade i
            - Captura tudo que é constante no tempo
            - Pode ser correlacionado com x_it
            
            **Quando usar:**
            - Interesse em efeito **dentro** das entidades
            - α_i pode estar correlacionado com regressores
            - Amostra é a população de interesse
            """)
        
        with col2:
            # Visualização FE
            np.random.seed(42)
            n_ent = 4
            n_per = 15
            
            fig = go.Figure()
            
            for i in range(n_ent):
                alpha = 5 + i * 3  # Interceptos diferentes
                x = np.linspace(0, 10, n_per) + np.random.normal(0, 0.5, n_per)
                y = alpha + 0.5 * x + np.random.normal(0, 0.5, n_per)
                
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'Entidade {i+1}'))
                
                # Linha de regressão para cada entidade
                x_line = np.linspace(x.min(), x.max(), 10)
                fig.add_trace(go.Scatter(x=x_line, y=alpha + 0.5 * x_line, 
                                        mode='lines', line=dict(dash='dash'),
                                        showlegend=False))
            
            fig.update_layout(title="FE: Interceptos Diferentes, Mesma Inclinação",
                             xaxis_title="X", yaxis_title="Y", height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Efeitos Aleatórios (RE)")
            
            st.markdown("""
            **Intuição:**
            > "Entidades são sorteios aleatórios de uma população maior"
            
            **Modelo:**
            $$y_{it} = \\alpha + u_i + \\beta x_{it} + \\varepsilon_{it}$$
            
            - u_i ~ N(0, σ²_u) = componente aleatório
            - u_i **não pode** estar correlacionado com x_it
            - Usa informação entre E dentro das entidades
            
            **Quando usar:**
            - Entidades são amostra de população maior
            - u_i não correlacionado com regressores
            - Interesse em efeito médio da população
            """)
        
        with col2:
            # Visualização RE
            fig = go.Figure()
            
            alpha_comum = 10
            for i in range(n_ent):
                u = np.random.normal(0, 2)  # Efeito aleatório
                x = np.linspace(0, 10, n_per) + np.random.normal(0, 0.5, n_per)
                y = alpha_comum + u + 0.5 * x + np.random.normal(0, 0.5, n_per)
                
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'Entidade {i+1}'))
            
            # Linha de regressão média
            x_line = np.linspace(0, 12, 50)
            fig.add_trace(go.Scatter(x=x_line, y=alpha_comum + 0.5 * x_line, 
                                    mode='lines', line=dict(color='black', width=2),
                                    name='Média Populacional'))
            
            fig.update_layout(title="RE: Variação em Torno da Média Populacional",
                             xaxis_title="X", yaxis_title="Y", height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    # Comparação com dados
    st.subheader("Exemplo: Produtividade Bancária")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fe_variance = st.slider("Variância dos efeitos fixos", 0.5, 5.0, 2.0, 0.5, key="fe_var")
        
        df = make_panel_data(n_entities=10, n_periods=15, beta=0.5, 
                            fe_variance=fe_variance, seed=42)
        
        # Estimar ambos
        pooled = fit_pooled_ols(df)
        fe = fit_fe_simple(df)
        re = fit_re_simple(df)
        
        st.markdown("**β verdadeiro:** 0.50")
    
    with col2:
        results_df = pd.DataFrame({
            'Modelo': ['Pooled OLS', 'Efeitos Fixos', 'Efeitos Aleatórios'],
            'β estimado': [f"{pooled['beta']:.3f}", f"{fe['beta']:.3f}", f"{re['beta']:.3f}"],
            'Erro Padrão': [f"{pooled['se_beta']:.3f}", f"{fe['se_beta']:.3f}", f"{re['se_beta']:.3f}"]
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        st.info(f"""
        💡 **Observação:** Com variância de FE = {fe_variance:.1f}:
        - Pooled OLS pode estar enviesado se α_i correlacionado com x
        - FE controla para diferenças entre bancos
        - RE é mais eficiente se suposições válidas
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - FE: Quando quer controlar para características fixas de cada unidade
    - RE: Quando interesse é em generalizar para população maior
    - Usa Hausman para decidir (próxima seção!)
    """)


def render_section_S6():
    """S6: Escolha do Modelo: Teste de Hausman"""
    st.header("🧪 Teste de Hausman: FE ou RE?")
    
    st.markdown("""
    O **teste de Hausman** ajuda a escolher entre Efeitos Fixos e Aleatórios.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Lógica do Teste")
        
        st.markdown("""
        **A pergunta-chave:**
        > "Os efeitos individuais estão correlacionados com as variáveis explicativas?"
        
        **Hipóteses:**
        - H₀: Corr(u_i, x_it) = 0 → RE é consistente E eficiente
        - H₁: Corr(u_i, x_it) ≠ 0 → RE é inconsistente (usar FE)
        
        **Estatística:**
        $$H = (\\beta_{FE} - \\beta_{RE})' [Var(\\beta_{FE}) - Var(\\beta_{RE})]^{-1} (\\beta_{FE} - \\beta_{RE})$$
        
        **Decisão:**
        - H grande (p < 0.05): Rejeita H₀ → Use FE
        - H pequeno (p ≥ 0.05): Não rejeita H₀ → Use RE
        """)
        
        st.subheader("Trade-off")
        
        st.markdown("""
        | Critério | FE | RE |
        |----------|-----|-----|
        | Consistência | Sempre ✓ | Só se H₀ ✓ |
        | Eficiência | Menor | Maior |
        | Usa info between | Não | Sim |
        """)
    
    with col2:
        st.subheader("Simulação Interativa")
        
        correlation = st.slider("Correlação α_i com x", 0.0, 0.8, 0.4, 0.1, key="corr_hausman")
        
        # Gerar dados com correlação controlada
        np.random.seed(42)
        n_entities = 15
        n_periods = 10
        
        # Efeitos fixos
        alpha = np.random.normal(0, 2, n_entities)
        
        data = []
        for i in range(n_entities):
            for t in range(n_periods):
                # X correlacionado com alpha_i
                x = 5 + correlation * alpha[i] + np.random.normal(0, 1)
                epsilon = np.random.normal(0, 1)
                y = alpha[i] + 0.5 * x + epsilon
                
                data.append({
                    'entity_id': i,
                    'entity': f'E_{i}',
                    'period': t,
                    'x': x,
                    'y': y
                })
        
        df = pd.DataFrame(data)
        
        # Teste de Hausman
        hausman = hausman_test_simple(df)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Estatística H", f"{hausman['H_stat']:.2f}")
        col_m2.metric("p-valor", f"{hausman['p_value']:.4f}")
        
        st.metric("β (FE)", f"{hausman['beta_fe']:.3f}")
        st.metric("β (RE)", f"{hausman['beta_re']:.3f}")
        
        if hausman['p_value'] < 0.05:
            st.error(f"🔴 **Recomendação: Use EFEITOS FIXOS**\n\np < 0.05 → Rejeita H₀ → RE inconsistente")
        else:
            st.success(f"🟢 **Recomendação: Use EFEITOS ALEATÓRIOS**\n\np ≥ 0.05 → Não rejeita H₀ → RE é preferível")
    
    # Quiz
    st.subheader("🧪 Quiz")
    
    st.markdown("""
    Um pesquisador estima um modelo de painel para 50 empresas em 10 anos.
    O teste de Hausman dá H = 15.3, p-valor = 0.001.
    """)
    
    resposta = st.radio(
        "Qual modelo deve usar?",
        ["Efeitos Aleatórios (mais eficiente)",
         "Efeitos Fixos (mais consistente)",
         "Pooled OLS (mais simples)"],
        key="quiz_hausman"
    )
    
    if st.button("Ver resposta", key="btn_hausman"):
        if resposta == "Efeitos Fixos (mais consistente)":
            st.success("""
            ✅ **Correto!**
            
            p = 0.001 < 0.05 → Rejeita H₀
            
            Há evidência de correlação entre efeitos individuais e regressores.
            RE seria inconsistente. FE é a escolha segura.
            """)
        else:
            st.error("Com p < 0.05, rejeitamos H₀. RE seria inconsistente. Use FE!")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Sempre roda teste de Hausman antes de interpretar
    - Se p < 0.05: usa FE (seguro, mesmo que menos eficiente)
    - Se p ≥ 0.05: pode usar RE (mais eficiente)
    """)


def render_section_S7():
    """S7: Estacionariedade e Longo Prazo em Painel"""
    st.header("📈 Estacionariedade e Cointegração em Painel")
    
    st.markdown("""
    Com painéis "macro" (muitos períodos), precisamos considerar não-estacionaridade.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Por que Importa?")
        
        st.markdown("""
        **Problemas de séries não-estacionárias:**
        - Regressão espúria (correlação sem sentido)
        - Inferência inválida
        - Estimativas inconsistentes
        
        **Em painel:**
        - Combina N séries temporais
        - Mais poder para detectar raiz unitária
        - Testes específicos: LLC, IPS, Fisher
        """)
        
        st.subheader("Testes de Raiz Unitária em Painel")
        
        st.markdown("""
        **LLC (Levin-Lin-Chu):**
        - Assume raiz unitária comum
        - H₀: Todas as séries têm RU
        - H₁: Todas estacionárias
        
        **IPS (Im-Pesaran-Shin):**
        - Permite heterogeneidade
        - H₀: Todas têm RU
        - H₁: Algumas estacionárias
        """)
    
    with col2:
        # Simular dados e testar
        df_credit = make_credit_case_data(n_countries=15, n_years=20, seed=42)
        
        # Teste simplificado
        ur_test = panel_unit_root_test_simple(df_credit, variable='log_gdp')
        coint_test = panel_cointegration_test_simple(df_credit)
        
        st.markdown("**Resultados dos Testes (Simulados):**")
        
        st.markdown("**Teste de Raiz Unitária (tipo LLC):**")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Estatística média", f"{ur_test['avg_stat']:.2f}")
        col_m2.metric("p-valor", f"{ur_test['p_value']:.4f}")
        
        st.markdown("**Teste de Cointegração em Painel:**")
        col_m3, col_m4 = st.columns(2)
        col_m3.metric("Estatística t", f"{coint_test['t_stat']:.2f}")
        col_m4.metric("Crítico 5%", f"{coint_test['critical_5']:.2f}")
        
        if coint_test['cointegrated']:
            st.success("✅ Evidência de cointegração: PIB e Crédito têm relação de longo prazo")
        else:
            st.warning("⚠️ Sem evidência forte de cointegração")
    
    with st.expander("📖 Cointegração em Painel: Aplicações"):
        st.markdown("""
        **Aplicações em finanças e macro:**
        
        1. **Crescimento e Desenvolvimento Financeiro:**
           - Crédito/PIB e PIB per capita
           - Relação de longo prazo entre países
        
        2. **Paridade de Poder de Compra (PPP):**
           - Câmbio e preços relativos
           - Testar se PPP vale no longo prazo
        
        3. **Estrutura de Capital:**
           - Alavancagem e características da firma
           - Velocidade de ajuste ao target
        
        **VECM em Painel:**
        - Combina cointegração com ajuste de curto prazo
        - Permite estimar velocidade de convergência
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Verifica estacionaridade em painéis longos (T > 20)
    - Usa cointegração para relações de longo prazo
    - Considera VECM para dinâmica de ajuste
    """)


def render_section_S8():
    """S8: Casos Práticos e Interpretação"""
    st.header("💼 Casos Práticos")
    
    tab1, tab2 = st.tabs(["🏦 Caso 1: Competição Bancária", "📉 Caso 2: Crises e Crédito"])
    
    with tab1:
        st.subheader("Caso: Determinantes da Rentabilidade Bancária")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Pergunta de pesquisa:**
            > "Quais fatores explicam a rentabilidade (ROE) dos bancos?"
            
            **Dados:**
            - 15 bancos, 10 anos
            - Variável dependente: ROE (%)
            - Regressores: Market Share, Cost/Income, Crescimento de Crédito, NPL
            
            **Modelo:**
            $$ROE_{it} = \\alpha_i + \\beta_1 MS_{it} + \\beta_2 CI_{it} + \\beta_3 CG_{it} + \\beta_4 NPL_{it} + \\varepsilon_{it}$$
            """)
        
        with col2:
            df_bank = make_banking_case_data(n_banks=15, n_years=10, seed=42)
            
            # Preparar para estimação
            df_bank['entity_id'] = df_bank['banco_id']
            df_bank['entity'] = df_bank['banco']
            df_bank['x'] = df_bank['market_share']  # Simplificado
            df_bank['y'] = df_bank['roe']
            
            # Estimar
            fe = fit_fe_simple(df_bank)
            re = fit_re_simple(df_bank)
            hausman = hausman_test_simple(df_bank)
            
            st.dataframe(df_bank.head(5)[['banco', 'ano', 'roe', 'market_share', 'cost_income']], 
                        use_container_width=True)
        
        st.subheader("Resultados da Estimação")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Efeitos Fixos:**")
            st.metric("β (Market Share)", f"{fe['beta']:.3f}")
            st.metric("R² within", f"{fe['r_squared_within']:.3f}")
        
        with col2:
            st.markdown("**Efeitos Aleatórios:**")
            st.metric("β (Market Share)", f"{re['beta']:.3f}")
            st.metric("θ (peso within)", f"{re['theta']:.3f}")
        
        with col3:
            st.markdown("**Teste de Hausman:**")
            st.metric("H-stat", f"{hausman['H_stat']:.2f}")
            st.metric("p-valor", f"{hausman['p_value']:.4f}")
            st.markdown(f"**Recomendação:** {hausman['recommendation']}")
        
        with st.expander("📖 Interpretação dos Resultados"):
            st.markdown(f"""
            **Leitura da tabela:**
            
            - β = {fe['beta']:.3f}: Aumento de 1 p.p. em market share está associado 
              a variação de {fe['beta']:.2f} p.p. no ROE (controlando por efeitos fixos)
            
            - R² within = {fe['r_squared_within']:.3f}: Variação **dentro** de cada banco 
              explica {fe['r_squared_within']*100:.1f}% da variação do ROE
            
            - Teste de Hausman p = {hausman['p_value']:.4f}: 
              {'Rejeita H₀ → Use FE' if hausman['p_value'] < 0.05 else 'Não rejeita H₀ → RE é válido'}
            
            **Implicação gerencial:**
            {"Ganhar market share está associado a maior rentabilidade, mesmo controlando para características fixas do banco." if fe['beta'] > 0 else "Não há evidência de que market share aumenta rentabilidade."}
            """)
    
    with tab2:
        st.subheader("Caso: Crédito e Crescimento em Crises")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Pergunta de pesquisa:**
            > "Qual a relação entre crédito e crescimento econômico? 
            > Crises afetam essa relação?"
            
            **Dados:**
            - 20 países, 15 anos
            - Variáveis: log(PIB), log(Crédito), Crédito/PIB
            - Período inclui crise (anos 5-6 para metade dos países)
            
            **Análise:**
            - Cointegração entre crédito e PIB
            - Efeito diferencial em países com crise
            """)
        
        with col2:
            df_credit = make_credit_case_data(n_countries=20, n_years=15, seed=42)
            
            # Visualização
            fig = px.line(df_credit, x='ano', y='credit_gdp_ratio', color='pais',
                         title="Crédito/PIB por País ao Longo do Tempo")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparação antes/durante/depois da crise
        st.subheader("Análise: Impacto da Crise")
        
        # Identificar países com crise
        crisis_countries = df_credit[df_credit['pais_id'] < 10]['pais'].unique()
        
        df_credit['has_crisis'] = df_credit['pais'].isin(crisis_countries)
        df_credit['period'] = pd.cut(df_credit['ano'], 
                                     bins=[2004, 2009, 2011, 2020],
                                     labels=['Pré-Crise', 'Crise', 'Pós-Crise'])
        
        # Média de crescimento por grupo
        summary = df_credit.groupby(['has_crisis', 'period'])['gdp_growth'].mean().reset_index()
        
        fig = px.bar(summary, x='period', y='gdp_growth', color='has_crisis',
                    barmode='group', title="Crescimento Médio por Período",
                    labels={'gdp_growth': 'Crescimento (%)', 'period': 'Período',
                           'has_crisis': 'País com Crise'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretação:**
        - Países com crise tiveram queda significativa no crescimento (anos 5-6)
        - Recuperação parcial no período pós-crise
        - Painel permite comparar trajetórias (diff-in-diff implícito)
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Interpreta coeficientes em termos econômicos
    - Considera se efeito é dentro ou entre entidades
    - Usa resultados para decisões de estratégia e política
    """)


def render_section_S9():
    """S9: Resumo Executivo e Encerramento do Curso"""
    st.header("📋 Resumo Executivo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### O que Aprendemos sobre Dados em Painel
        
        ✅ **O que são painéis:**
        - Mesmas entidades observadas ao longo do tempo
        - Combinam cross-section e séries temporais
        - N × T observações = mais poder estatístico
        
        ✅ **Vantagens:**
        - Controlam heterogeneidade não observada
        - Reduzem viés de variável omitida
        - Capturam dinâmica temporal
        
        ✅ **Estrutura:**
        - Balanceado: todas obs presentes
        - Não balanceado: alguns missings (verificar padrão!)
        
        ✅ **SUR:**
        - Equações com erros correlacionados
        - Ganho de eficiência quando ρ alto
        
        ✅ **Efeitos Fixos vs Aleatórios:**
        - FE: intercepto específico por entidade (sempre consistente)
        - RE: efeito aleatório de população (mais eficiente se válido)
        - Hausman decide: p < 0.05 → FE
        
        ✅ **Longo prazo:**
        - Testar raiz unitária em painéis longos
        - Cointegração para relações de equilíbrio
        
        ✅ **Aplicações:**
        - Competição e rentabilidade bancária
        - Crédito e crescimento
        - Políticas e seus efeitos
        """)
    
    with col2:
        st.markdown("### 💡 Mensagem-Chave")
        
        st.info("""
        **"Painéis transformam dados em histórias de decisão"**
        
        Ao observar as mesmas unidades ao longo do tempo:
        - Controlamos o que não muda
        - Isolamos o que muda
        - Estimamos efeitos causais com mais confiança
        """)
        
        st.markdown("### 🧪 Quiz Final")
        
        resposta = st.radio(
            "Qual a principal vantagem de FE sobre Pooled OLS?",
            ["Mais observações",
             "Controla heterogeneidade não observada constante",
             "Estimativas mais eficientes"],
            key="quiz_final"
        )
        
        if st.button("Ver resposta", key="btn_final"):
            if resposta == "Controla heterogeneidade não observada constante":
                st.success("""
                ✅ **Correto!**
                
                FE adiciona um intercepto para cada entidade, 
                absorvendo tudo que é constante no tempo.
                
                Isso elimina viés de variáveis omitidas 
                que são fixas por entidade.
                """)
            else:
                st.error("FE controla para características fixas de cada entidade que poderiam viesar OLS.")
    
    st.markdown("---")
    
    st.subheader("🎓 Encerramento do Curso")
    
    st.markdown("""
    ### Integração dos Módulos
    
    Este curso cobriu a jornada completa da econometria aplicada:
    """)
    
    modules = pd.DataFrame({
        'Módulo': ['1-2', '3-4', '5', '6', '7', '8', '9', '10'],
        'Tema': ['Fundamentos e CLRM', 'Diagnóstico e Correções', 'Causalidade',
                'Séries Univariadas', 'Modelos Multivariados', 'Cointegração',
                'Volatilidade', 'Dados em Painel'],
        'Aplicação': ['Base teórica', 'Validação de modelos', 'Decisões causais',
                     'Previsão', 'Sistemas e VAR', 'Longo prazo', 'Risco', 'Comparações']
    })
    st.dataframe(modules, use_container_width=True, hide_index=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔧 Ferramentas:**
        - OLS e extensões
        - Diagnósticos
        - IV/2SLS
        - ARIMA, VAR, VECM
        - GARCH, DCC
        - FE, RE, Hausman
        """)
    
    with col2:
        st.markdown("""
        **📊 Dados:**
        - Cross-section
        - Séries temporais
        - Painéis
        - Balanceados/não
        """)
    
    with col3:
        st.markdown("""
        **💼 Decisões:**
        - Previsão
        - Causalidade
        - Risco
        - Política
        - Estratégia
        """)
    
    st.success("""
    🎓 **Parabéns!** Você completou o Laboratório de Econometria.
    
    Agora você tem as ferramentas para:
    - Analisar dados com rigor metodológico
    - Escolher o modelo adequado para cada situação
    - Interpretar resultados para tomada de decisão
    - Comunicar achados com confiança
    
    **Continue praticando com dados reais!**
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com tudo isso?**
    - Aplica a ferramenta certa para cada problema
    - Interpreta resultados considerando limitações
    - Comunica incerteza junto com conclusões
    - Usa evidência para melhorar decisões
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("📊 Módulo 10: Dados em Painel")
    st.markdown("**Laboratório de Econometria** | FE, RE, Hausman e Aplicações")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Explicar o que são **dados em painel** e suas vantagens
        - Mostrar diferença entre **balanceado** e **não balanceado**
        - Introduzir **SUR** para erros correlacionados
        - Diferenciar **Efeitos Fixos** e **Aleatórios**
        - Ensinar o **Teste de Hausman** para escolha do modelo
        - Apresentar **estacionariedade** e **cointegração** em painel
        - Aplicar em **casos práticos** do setor bancário
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "📊 Introdução",
        "S2": "💪 Vantagens",
        "S3": "📋 Balanceado vs Não",
        "S4": "🔗 SUR",
        "S5": "⚖️ FE vs RE",
        "S6": "🧪 Hausman",
        "S7": "📈 Longo Prazo",
        "S8": "💼 Casos Práticos",
        "S9": "📋 Resumo"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.success("""
    🎓 **Último Módulo!**
    
    Painéis combinam o melhor de 
    cross-section e séries temporais.
    """)
    
    # Renderizar seção selecionada
    if secao_selecionada == "S1":
        render_section_S1()
    elif secao_selecionada == "S2":
        render_section_S2()
    elif secao_selecionada == "S3":
        render_section_S3()
    elif secao_selecionada == "S4":
        render_section_S4()
    elif secao_selecionada == "S5":
        render_section_S5()
    elif secao_selecionada == "S6":
        render_section_S6()
    elif secao_selecionada == "S7":
        render_section_S7()
    elif secao_selecionada == "S8":
        render_section_S8()
    elif secao_selecionada == "S9":
        render_section_S9()


# =============================================================================
# EXECUÇÃO STANDALONE (para testes)
# =============================================================================

if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Módulo 10: Dados em Painel",
            page_icon="📊",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()