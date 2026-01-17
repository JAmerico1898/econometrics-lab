"""
Laboratório de Econometria - Module 9: Modelling Volatility and Correlation
Aplicativo educacional interativo para GARCH, volatilidade condicional e correlação dinâmica.
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

def simulate_returns_stylized(n: int = 500, garch_alpha: float = 0.1, 
                              garch_beta: float = 0.85, seed: int = 42) -> dict:
    """
    Simula retornos com características estilizadas:
    - Volatility clustering
    - Caudas pesadas
    - Efeito alavancagem (assimetria)
    """
    np.random.seed(seed)
    
    # Simular GARCH(1,1) com assimetria
    omega = 0.00001
    alpha = garch_alpha
    beta = garch_beta
    gamma = 0.05  # Assimetria
    
    h = np.zeros(n)
    r = np.zeros(n)
    z = np.random.standard_t(df=5, size=n)  # Caudas pesadas
    
    h[0] = omega / (1 - alpha - beta)
    r[0] = np.sqrt(h[0]) * z[0]
    
    for t in range(1, n):
        # GJR-GARCH
        leverage = gamma * (r[t-1] < 0) * r[t-1]**2
        h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1] + leverage
        h[t] = max(h[t], 1e-8)
        r[t] = np.sqrt(h[t]) * z[t]
    
    # Escalar para % (retornos diários típicos)
    r = r * 100
    
    return {
        'returns': r,
        'variance': h * 10000,
        'volatility': np.sqrt(h) * 100
    }


def compute_hist_vol(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Calcula volatilidade histórica com janela móvel."""
    n = len(returns)
    vol = np.full(n, np.nan)
    
    for t in range(window, n):
        vol[t] = np.std(returns[t-window:t], ddof=1)
    
    return vol


def compute_ewma_vol(returns: np.ndarray, lambd: float = 0.94) -> np.ndarray:
    """Calcula volatilidade EWMA (RiskMetrics)."""
    n = len(returns)
    var = np.zeros(n)
    
    # Inicializar com variância amostral dos primeiros 20 obs
    var[0] = np.var(returns[:min(20, n)])
    
    for t in range(1, n):
        var[t] = lambd * var[t-1] + (1 - lambd) * returns[t-1]**2
    
    return np.sqrt(var)


def simulate_garch(n: int = 300, omega: float = 0.00001, alpha: float = 0.1, 
                   beta: float = 0.85, seed: int = 42) -> dict:
    """Simula GARCH(1,1) puro."""
    np.random.seed(seed)
    
    h = np.zeros(n)
    r = np.zeros(n)
    z = np.random.normal(0, 1, n)
    
    # Variância incondicional
    if alpha + beta < 1:
        h[0] = omega / (1 - alpha - beta)
    else:
        h[0] = omega
    
    r[0] = np.sqrt(h[0]) * z[0]
    
    for t in range(1, n):
        h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
        h[t] = max(h[t], 1e-10)
        r[t] = np.sqrt(h[t]) * z[t]
    
    # Escalar
    r = r * 100
    h = h * 10000
    
    return {
        'returns': r,
        'variance': h,
        'volatility': np.sqrt(h)
    }


def fit_garch_mle_simple(returns: np.ndarray, omega_init: float = 0.01,
                         alpha_init: float = 0.1, beta_init: float = 0.8) -> dict:
    """
    Ajusta GARCH(1,1) por máxima verossimilhança (simplificado).
    Usa grid search para demonstração didática.
    """
    r = returns / 100  # Desescalar
    n = len(r)
    
    best_ll = -np.inf
    best_params = None
    
    # Grid search simplificado
    for alpha in np.arange(0.02, 0.25, 0.02):
        for beta in np.arange(0.7, 0.95, 0.02):
            if alpha + beta >= 0.999:
                continue
            
            omega = np.var(r) * (1 - alpha - beta)
            omega = max(omega, 1e-8)
            
            # Calcular variância condicional
            h = np.zeros(n)
            h[0] = np.var(r)
            
            for t in range(1, n):
                h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
                h[t] = max(h[t], 1e-10)
            
            # Log-verossimilhança (normal)
            ll = -0.5 * np.sum(np.log(h) + r**2 / h)
            
            if ll > best_ll:
                best_ll = ll
                best_params = {'omega': omega, 'alpha': alpha, 'beta': beta}
    
    if best_params is None:
        best_params = {'omega': omega_init, 'alpha': alpha_init, 'beta': beta_init}
        best_ll = -np.inf
    
    # Calcular variância condicional com melhores parâmetros
    h = np.zeros(n)
    h[0] = np.var(r)
    for t in range(1, n):
        h[t] = best_params['omega'] + best_params['alpha'] * r[t-1]**2 + best_params['beta'] * h[t-1]
        h[t] = max(h[t], 1e-10)
    
    return {
        'omega': best_params['omega'],
        'alpha': best_params['alpha'],
        'beta': best_params['beta'],
        'persistence': best_params['alpha'] + best_params['beta'],
        'log_likelihood': best_ll,
        'variance': h * 10000,
        'volatility': np.sqrt(h) * 100
    }


def arch_effects_test(returns: np.ndarray, lags: int = 5) -> dict:
    """
    Teste de efeitos ARCH (Engle's ARCH-LM test).
    Regressa resíduos² em seus lags e testa significância conjunta.
    """
    r2 = returns**2
    n = len(r2)
    
    # Construir matriz de lags
    y = r2[lags:]
    X = np.column_stack([np.ones(n - lags)] + [r2[lags-i-1:n-i-1] for i in range(lags)])
    
    # OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat
    
    # R²
    sse = np.sum(residuals**2)
    sst = np.sum((y - np.mean(y))**2)
    r_squared = 1 - sse / sst
    
    # Estatística LM = n * R²
    lm_stat = (n - lags) * r_squared
    p_value = 1 - stats.chi2.cdf(lm_stat, lags)
    
    return {
        'lm_stat': lm_stat,
        'p_value': p_value,
        'r_squared': r_squared,
        'lags': lags
    }


def simulate_asymmetric_garch(n: int = 300, omega: float = 0.00001, alpha: float = 0.05,
                               beta: float = 0.85, gamma: float = 0.1, 
                               model: str = 'GJR', seed: int = 42) -> dict:
    """Simula GJR-GARCH ou EGARCH com assimetria."""
    np.random.seed(seed)
    
    h = np.zeros(n)
    r = np.zeros(n)
    z = np.random.normal(0, 1, n)
    
    h[0] = omega / (1 - alpha - beta - gamma/2) if (alpha + beta + gamma/2) < 1 else omega * 10
    r[0] = np.sqrt(h[0]) * z[0]
    
    for t in range(1, n):
        if model == 'GJR':
            # GJR-GARCH: termo adicional se retorno negativo
            indicator = 1 if r[t-1] < 0 else 0
            h[t] = omega + alpha * r[t-1]**2 + gamma * indicator * r[t-1]**2 + beta * h[t-1]
        else:
            # EGARCH simplificado (em variância, não log)
            shock = np.abs(r[t-1]) / np.sqrt(max(h[t-1], 1e-10))
            asym = gamma * r[t-1] / np.sqrt(max(h[t-1], 1e-10))
            h[t] = omega + alpha * shock**2 * h[t-1] + beta * h[t-1] + asym * h[t-1]
        
        h[t] = max(h[t], 1e-10)
        r[t] = np.sqrt(h[t]) * z[t]
    
    r = r * 100
    h = h * 10000
    
    return {
        'returns': r,
        'variance': h,
        'volatility': np.sqrt(h)
    }


def news_impact_curve(omega: float, alpha: float, beta: float, gamma: float = 0.0,
                      h_prev: float = 1.0) -> dict:
    """Calcula curva de impacto de notícias (News Impact Curve)."""
    shocks = np.linspace(-3, 3, 100)
    
    # GARCH simétrico
    h_symmetric = omega + alpha * shocks**2 + beta * h_prev
    
    # GJR assimétrico
    h_asymmetric = np.where(
        shocks < 0,
        omega + (alpha + gamma) * shocks**2 + beta * h_prev,
        omega + alpha * shocks**2 + beta * h_prev
    )
    
    return {
        'shocks': shocks,
        'h_symmetric': h_symmetric,
        'h_asymmetric': h_asymmetric
    }


def simulate_time_varying_corr(n: int = 300, base_corr: float = 0.5, 
                               crisis_corr: float = 0.9, crisis_start: int = 150,
                               crisis_end: int = 200, seed: int = 42) -> dict:
    """Simula dois ativos com correlação que muda no tempo."""
    np.random.seed(seed)
    
    # Correlação variante no tempo
    rho = np.full(n, base_corr)
    rho[crisis_start:crisis_end] = crisis_corr
    
    # Suavizar transição
    for t in range(crisis_start, min(crisis_start + 10, n)):
        rho[t] = base_corr + (crisis_corr - base_corr) * (t - crisis_start) / 10
    for t in range(crisis_end - 10, crisis_end):
        rho[t] = crisis_corr - (crisis_corr - base_corr) * (crisis_end - t) / 10
    
    # Volatilidades
    vol1 = np.full(n, 0.01)
    vol2 = np.full(n, 0.015)
    vol1[crisis_start:crisis_end] = 0.025
    vol2[crisis_start:crisis_end] = 0.03
    
    # Gerar retornos correlacionados
    r1 = np.zeros(n)
    r2 = np.zeros(n)
    
    for t in range(n):
        z1 = np.random.normal()
        z2 = rho[t] * z1 + np.sqrt(1 - rho[t]**2) * np.random.normal()
        r1[t] = vol1[t] * z1
        r2[t] = vol2[t] * z2
    
    return {
        'r1': r1 * 100,
        'r2': r2 * 100,
        'vol1': vol1 * 100,
        'vol2': vol2 * 100,
        'true_corr': rho
    }


def compute_dcc_proxy(r1: np.ndarray, r2: np.ndarray, lambd: float = 0.94) -> np.ndarray:
    """
    Aproximação didática de DCC usando EWMA para covariância.
    """
    n = len(r1)
    
    # EWMA para variâncias
    var1 = compute_ewma_vol(r1, lambd)**2
    var2 = compute_ewma_vol(r2, lambd)**2
    
    # EWMA para covariância
    cov = np.zeros(n)
    cov[0] = np.cov(r1[:20], r2[:20])[0, 1] if n > 20 else r1[0] * r2[0]
    
    for t in range(1, n):
        cov[t] = lambd * cov[t-1] + (1 - lambd) * r1[t-1] * r2[t-1]
    
    # Correlação condicional
    corr = cov / (np.sqrt(var1) * np.sqrt(var2) + 1e-10)
    corr = np.clip(corr, -0.999, 0.999)
    
    return corr


def compute_dynamic_hedge_ratio(r_asset: np.ndarray, r_hedge: np.ndarray, 
                                 lambd: float = 0.94) -> np.ndarray:
    """Calcula hedge ratio dinâmico: h = Cov(asset, hedge) / Var(hedge)."""
    n = len(r_asset)
    
    # EWMA para variância do hedge
    var_hedge = compute_ewma_vol(r_hedge, lambd)**2
    
    # EWMA para covariância
    cov = np.zeros(n)
    cov[0] = np.cov(r_asset[:20], r_hedge[:20])[0, 1] if n > 20 else 0
    
    for t in range(1, n):
        cov[t] = lambd * cov[t-1] + (1 - lambd) * r_asset[t-1] * r_hedge[t-1]
    
    # Hedge ratio
    h = cov / (var_hedge + 1e-10)
    
    return h


def compute_time_varying_beta(r_asset: np.ndarray, r_market: np.ndarray,
                               lambd: float = 0.94) -> np.ndarray:
    """Calcula beta variante no tempo usando EWMA."""
    return compute_dynamic_hedge_ratio(r_asset, r_market, lambd)


def compute_var_models(returns: np.ndarray, confidence: float = 0.95) -> dict:
    """Calcula VaR usando diferentes métodos."""
    n = len(returns)
    
    # VaR Histórico (janela de 60 dias)
    var_hist = np.full(n, np.nan)
    window = 60
    for t in range(window, n):
        var_hist[t] = -np.percentile(returns[t-window:t], (1 - confidence) * 100)
    
    # VaR EWMA
    vol_ewma = compute_ewma_vol(returns, 0.94)
    z_score = stats.norm.ppf(confidence)
    var_ewma = vol_ewma * z_score
    
    # VaR GARCH (simplificado - usando EWMA como proxy)
    var_garch = var_ewma * 1.05  # Ajuste para simular diferença
    
    return {
        'var_hist': var_hist,
        'var_ewma': var_ewma,
        'var_garch': var_garch
    }


def backtest_var_exceedances(returns: np.ndarray, var: np.ndarray) -> dict:
    """Backtest de VaR: conta violações."""
    # Violação quando perda > VaR
    violations = (-returns) > var
    
    # Desconsiderar NaN
    valid = ~np.isnan(var)
    n_valid = np.sum(valid)
    n_violations = np.sum(violations & valid)
    
    violation_rate = n_violations / n_valid if n_valid > 0 else 0
    
    return {
        'n_violations': n_violations,
        'n_observations': n_valid,
        'violation_rate': violation_rate,
        'violations': violations
    }


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: Por que a volatilidade importa? (fatos estilizados)"""
    st.header("📊 Por que a Volatilidade Importa?")
    
    st.markdown("""
    Modelos lineares com variância constante **falham em finanças** porque:
    - Volatilidade varia ao longo do tempo
    - Retornos têm caudas pesadas
    - Quedas geram mais volatilidade que altas
    """)
    
    tab1, tab2, tab3 = st.tabs(["📈 Clustering", "📊 Caudas Pesadas", "↘️ Alavancagem"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Volatility Clustering")
            
            st.markdown("""
            **Fato estilizado #1:**
            > "Grandes retornos tendem a ser seguidos por grandes retornos"
            
            Períodos de alta volatilidade se agrupam:
            - Crises: sequência de dias turbulentos
            - Calmaria: sequência de dias tranquilos
            
            **Implicação:** Volatilidade de ontem prevê volatilidade de hoje.
            """)
        
        with col2:
            data = simulate_returns_stylized(n=400, seed=42)
            
            fig = make_subplots(rows=2, cols=1,
                               subplot_titles=["Retornos", "Volatilidade Condicional"],
                               row_heights=[0.5, 0.5])
            
            fig.add_trace(go.Scatter(y=data['returns'], mode='lines',
                                    line=dict(width=0.8)), row=1, col=1)
            fig.add_trace(go.Scatter(y=data['volatility'], mode='lines',
                                    line=dict(color='red')), row=2, col=1)
            
            fig.update_layout(height=400, showlegend=False)
            fig.update_yaxes(title_text="Retorno (%)", row=1, col=1)
            fig.update_yaxes(title_text="Volatilidade (%)", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Caudas Pesadas (Leptocurtose)")
            
            st.markdown("""
            **Fato estilizado #2:**
            > "Eventos extremos são mais frequentes que a Normal prevê"
            
            - Distribuição Normal subestima crashes
            - Retornos têm curtose > 3 (leptocurtose)
            - VaR baseado em Normal é otimista demais
            
            **Implicação:** Modelos de risco devem considerar caudas pesadas.
            """)
            
            data = simulate_returns_stylized(n=1000, seed=42)
            curtose = stats.kurtosis(data['returns']) + 3
            st.metric("Curtose dos retornos", f"{curtose:.2f}", 
                     help="Normal = 3.0")
        
        with col2:
            # QQ Plot
            fig = go.Figure()
            
            # Quantis teóricos vs empíricos
            sorted_returns = np.sort(data['returns'])
            theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
            
            fig.add_trace(go.Scatter(x=theoretical, y=sorted_returns,
                                    mode='markers', name='Dados', 
                                    marker=dict(size=3, opacity=0.5)))
            
            # Linha 45°
            min_val = min(theoretical.min(), sorted_returns.min())
            max_val = max(theoretical.max(), sorted_returns.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                    mode='lines', name='Normal',
                                    line=dict(color='red', dash='dash')))
            
            fig.update_layout(
                title="QQ-Plot: Retornos vs Normal",
                xaxis_title="Quantis Teóricos (Normal)",
                yaxis_title="Quantis Empíricos",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Pontos afastados da linha = caudas mais pesadas que Normal")
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Efeito de Alavancagem")
            
            st.markdown("""
            **Fato estilizado #3:**
            > "Retornos negativos aumentam mais a volatilidade que positivos"
            
            **Por quê?**
            - Queda de preço → aumento de alavancagem (dívida/equity)
            - Maior alavancagem → maior risco
            - Também: feedback comportamental (pânico)
            
            **Implicação:** Modelos devem capturar assimetria (GJR, EGARCH).
            """)
        
        with col2:
            # Scatter: retorno vs volatilidade futura
            data = simulate_returns_stylized(n=500, seed=42)
            
            r_lag = data['returns'][:-1]
            vol_next = data['volatility'][1:]
            
            fig = px.scatter(x=r_lag, y=vol_next, opacity=0.5,
                            labels={'x': 'Retorno t', 'y': 'Volatilidade t+1'})
            
            # Linhas de tendência para cada lado
            neg_mask = r_lag < 0
            pos_mask = r_lag >= 0
            
            if np.sum(neg_mask) > 2:
                z_neg = np.polyfit(r_lag[neg_mask], vol_next[neg_mask], 1)
                x_neg = np.linspace(r_lag[neg_mask].min(), 0, 20)
                fig.add_trace(go.Scatter(x=x_neg, y=z_neg[0]*x_neg + z_neg[1],
                                        mode='lines', name='Negativos',
                                        line=dict(color='red')))
            
            if np.sum(pos_mask) > 2:
                z_pos = np.polyfit(r_lag[pos_mask], vol_next[pos_mask], 1)
                x_pos = np.linspace(0, r_lag[pos_mask].max(), 20)
                fig.add_trace(go.Scatter(x=x_pos, y=z_pos[0]*x_pos + z_pos[1],
                                        mode='lines', name='Positivos',
                                        line=dict(color='green')))
            
            fig.update_layout(title="Assimetria: Efeito Alavancagem", height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("💼 Conexão com Decisões"):
        st.markdown("""
        **VaR (Value-at-Risk):**
        - Caudas pesadas → VaR Normal subestima perdas extremas
        - Clustering → VaR deve variar no tempo
        
        **Opções (Black-Scholes):**
        - BS assume volatilidade constante
        - Smile de volatilidade: mercado precifica vol diferente por strike
        - Modelos GARCH melhoram precificação
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa modelos que capturam volatilidade variável
    - Não confia em VaR baseado em Normal
    - Monitora clustering para ajustar limites de risco
    """)


def render_section_S2():
    """S2: Da volatilidade histórica ao EWMA"""
    st.header("📏 Volatilidade Histórica vs EWMA")
    
    st.markdown("""
    Antes de GARCH, vamos comparar métodos mais simples:
    - **Histórica:** Desvio padrão de uma janela fixa
    - **EWMA:** Média ponderada exponencial (RiskMetrics)
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Parâmetros")
        
        window = st.slider("Janela histórica (dias)", 10, 60, 20, key="window_hist")
        lambd = st.slider("Lambda EWMA", 0.85, 0.99, 0.94, 0.01, key="lambda_ewma",
                         help="RiskMetrics usa λ=0.94")
        
        st.markdown("""
        **Volatilidade Histórica:**
        - Todos os dias na janela têm peso igual
        - Reage lentamente a choques
        - Demora para "esquecer" eventos antigos
        
        **EWMA:**
        - Pesos decrescem exponencialmente
        - Reage mais rápido a choques recentes
        - Não tem nível de "longo prazo"
        """)
    
    with col2:
        # Simular dados com um choque
        np.random.seed(42)
        n = 200
        returns = np.random.normal(0, 1, n)
        # Adicionar choque
        returns[100:110] = returns[100:110] * 4  # Período de alta vol
        
        vol_hist = compute_hist_vol(returns, window)
        vol_ewma = compute_ewma_vol(returns, lambd)
        
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=["Retornos (com choque)", "Volatilidade Estimada"],
                           row_heights=[0.4, 0.6])
        
        fig.add_trace(go.Scatter(y=returns, mode='lines', name='Retornos',
                                line=dict(width=0.8)), row=1, col=1)
        
        fig.add_trace(go.Scatter(y=vol_hist, mode='lines', name=f'Histórica ({window}d)',
                                line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(y=vol_ewma, mode='lines', name=f'EWMA (λ={lambd})',
                                line=dict(color='red')), row=2, col=1)
        
        # Marcar período de choque
        fig.add_vrect(x0=100, x1=110, fillcolor="yellow", opacity=0.2,
                     annotation_text="Choque", row=1, col=1)
        fig.add_vrect(x0=100, x1=110, fillcolor="yellow", opacity=0.2, row=2, col=1)
        
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True, key=f"vol_comp_{window}_{lambd}")
    
    # Comparação de VaR
    st.subheader("Impacto no VaR (95%)")
    
    z_95 = stats.norm.ppf(0.95)
    var_hist = vol_hist * z_95
    var_ewma = vol_ewma * z_95
    
    col1, col2, col3 = st.columns(3)
    col1.metric("VaR Histórico (média)", f"{np.nanmean(var_hist):.2f}%")
    col2.metric("VaR EWMA (média)", f"{np.nanmean(var_ewma):.2f}%")
    col3.metric("VaR EWMA no pico", f"{np.nanmax(var_ewma):.2f}%")
    
    with st.expander("📖 Limitações"):
        st.markdown("""
        **Histórica:**
        - Ghost effect: choque antigo continua afetando até sair da janela
        - Reação em degrau (não suave)
        
        **EWMA:**
        - Não tem reversão à média
        - Após choque, volatilidade só cai se retornos forem pequenos
        - Não captura bem a dinâmica de longo prazo
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - EWMA é melhor que histórica para reagir a mudanças
    - Mas GARCH é ainda melhor para capturar dinâmica completa
    """)


def render_section_S3():
    """S3: GARCH(1,1): risco que muda com o tempo"""
    st.header("📈 GARCH(1,1): Risco que Muda com o Tempo")
    
    st.markdown("""
    **GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)**
    combina memória (persistência) com reação a choques.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Intuição do GARCH(1,1)")
        
        st.markdown("""
        **Equação:**
        $$h_t = \\omega + \\alpha \\cdot r_{t-1}^2 + \\beta \\cdot h_{t-1}$$
        
        **Componentes:**
        - **ω (omega):** Nível base de variância
        - **α (alpha):** Reação ao choque de ontem
        - **β (beta):** Persistência (memória)
        
        **Persistência:** α + β
        - Próximo de 1 = alta memória
        - < 1 = reverte à média de longo prazo
        
        **Variância incondicional:**
        $$\\bar{h} = \\frac{\\omega}{1 - \\alpha - \\beta}$$
        """)
        
        st.subheader("Simulador")
        
        alpha = st.slider("α (choque)", 0.01, 0.3, 0.1, 0.01, key="alpha_garch")
        beta = st.slider("β (persistência)", 0.5, 0.95, 0.85, 0.01, key="beta_garch")
        
        persistence = alpha + beta
        
        if persistence >= 1:
            st.error(f"⚠️ α + β = {persistence:.2f} ≥ 1: Processo explosivo!")
        else:
            st.success(f"✅ Persistência: α + β = {persistence:.2f}")
    
    with col2:
        omega = 0.00001  # Fixo para simplicidade
        
        if alpha + beta < 1:
            data = simulate_garch(n=300, omega=omega, alpha=alpha, beta=beta, seed=42)
            
            fig = make_subplots(rows=2, cols=1,
                               subplot_titles=["Retornos Simulados", "Variância Condicional h_t"],
                               row_heights=[0.5, 0.5])
            
            fig.add_trace(go.Scatter(y=data['returns'], mode='lines',
                                    line=dict(width=0.8)), row=1, col=1)
            fig.add_trace(go.Scatter(y=data['variance'], mode='lines',
                                    line=dict(color='red')), row=2, col=1)
            
            # Variância incondicional
            h_bar = omega * 10000 / (1 - alpha - beta)
            fig.add_hline(y=h_bar, line_dash="dash", line_color="green",
                         annotation_text=f"h̄ = {h_bar:.4f}", row=2, col=1)
            
            fig.update_layout(height=450, showlegend=False)
            fig.update_yaxes(title_text="Retorno (%)", row=1, col=1)
            fig.update_yaxes(title_text="Variância", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True, key=f"garch_{alpha}_{beta}")
            
            # Meia-vida
            half_life = np.log(0.5) / np.log(persistence)
            st.metric("Meia-vida do choque", f"{half_life:.1f} dias",
                     help="Tempo para efeito de choque reduzir pela metade")
        else:
            st.warning("Ajuste os parâmetros para α + β < 1")
    
    # Quiz
    st.subheader("🧪 Quiz")
    
    st.markdown("Se α = 0.05 e β = 0.90, qual é a persistência e o que isso significa?")
    
    resposta = st.radio(
        "Selecione:",
        ["0.95 - volatilidade muda muito rápido",
         "0.95 - choques demoram muito para dissipar",
         "0.85 - volatilidade é praticamente constante"],
        key="quiz_garch"
    )
    
    if st.button("Ver resposta", key="btn_garch"):
        if resposta == "0.95 - choques demoram muito para dissipar":
            st.success("""
            ✅ **Correto!**
            
            Persistência = 0.05 + 0.90 = 0.95
            
            Alta persistência significa que choques têm efeito duradouro.
            Meia-vida ≈ 14 dias.
            """)
        else:
            st.error("A persistência é 0.95, e valores próximos de 1 indicam memória longa.")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa α + β para entender quão rápido risco muda
    - Alta persistência → ajustes de hedge mais frequentes
    - GARCH permite previsão de volatilidade para VaR e opções
    """)


def render_section_S4():
    """S4: Estimação e Diagnóstico (funciona mesmo?)"""
    st.header("🔧 Estimação e Diagnóstico")
    
    tab1, tab2, tab3 = st.tabs(["📊 Máxima Verossimilhança", "🧪 Teste ARCH", "✅ Checklist"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Máxima Verossimilhança (MLE)")
            
            st.markdown("""
            **Intuição:**
            > "Encontrar os parâmetros que tornam os dados observados mais prováveis"
            
            **Log-verossimilhança Normal:**
            $$\\ell = -\\frac{1}{2} \\sum_t \\left( \\log h_t + \\frac{r_t^2}{h_t} \\right)$$
            
            **Processo:**
            1. Chutar valores iniciais de ω, α, β
            2. Calcular h_t para toda a série
            3. Calcular log-verossimilhança
            4. Otimizar (maximizar ℓ)
            
            **Cuidado:** Ótimos locais! Resultado pode depender do ponto inicial.
            """)
            
            seed_data = st.slider("Seed dos dados", 1, 100, 42, key="seed_mle")
        
        with col2:
            # Simular e estimar
            true_alpha = 0.10
            true_beta = 0.85
            
            data = simulate_garch(n=500, alpha=true_alpha, beta=true_beta, seed=seed_data)
            
            # Estimar
            result = fit_garch_mle_simple(data['returns'])
            
            st.markdown("**Parâmetros Verdadeiros vs Estimados:**")
            
            comp_df = pd.DataFrame({
                'Parâmetro': ['α', 'β', 'Persistência'],
                'Verdadeiro': [true_alpha, true_beta, true_alpha + true_beta],
                'Estimado': [result['alpha'], result['beta'], result['persistence']]
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.metric("Log-Verossimilhança", f"{result['log_likelihood']:.1f}")
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=np.sqrt(data['variance']), mode='lines',
                                    name='Verdadeira'))
            fig.add_trace(go.Scatter(y=result['volatility'], mode='lines',
                                    name='Estimada', line=dict(dash='dash')))
            fig.update_layout(title="Volatilidade: Verdadeira vs Estimada", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Teste de Efeitos ARCH")
            
            st.markdown("""
            **Pergunta:** "Há heterocedasticidade condicional nos dados?"
            
            **Teste ARCH-LM (Engle):**
            1. Regressa r² contra seus lags
            2. Testa se coeficientes são conjuntamente zero
            
            **Hipóteses:**
            - H₀: Sem efeitos ARCH (variância constante)
            - H₁: Há efeitos ARCH (variância muda)
            
            **Decisão:**
            - p < 0.05: Rejeita H₀ → Use GARCH!
            - p ≥ 0.05: Não rejeita H₀ → GARCH pode ser overkill
            """)
            
            lags_test = st.slider("Número de lags", 1, 10, 5, key="lags_arch")
        
        with col2:
            # Testar nos dados simulados
            data = simulate_garch(n=500, alpha=0.1, beta=0.85, seed=42)
            arch_test = arch_effects_test(data['returns'], lags=lags_test)
            
            st.markdown("**Resultado do Teste ARCH-LM:**")
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("LM Statistic", f"{arch_test['lm_stat']:.2f}")
            col_m2.metric("p-valor", f"{arch_test['p_value']:.4f}")
            
            if arch_test['p_value'] < 0.05:
                st.success("✅ Rejeita H₀: Há efeitos ARCH — GARCH é justificado!")
            else:
                st.info("Não rejeita H₀: Talvez GARCH não seja necessário.")
            
            # Comparar com dados sem ARCH
            st.markdown("---")
            st.markdown("**Comparação: Dados com vs sem efeitos ARCH**")
            
            np.random.seed(42)
            returns_no_arch = np.random.normal(0, 1, 500)
            arch_test_no = arch_effects_test(returns_no_arch, lags=lags_test)
            
            comp_df = pd.DataFrame({
                'Dados': ['GARCH (com ARCH)', 'Normal (sem ARCH)'],
                'LM Stat': [arch_test['lm_stat'], arch_test_no['lm_stat']],
                'p-valor': [arch_test['p_value'], arch_test_no['p_value']]
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Checklist: Quando Usar GARCH?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ Use GARCH quando:**
            - Retornos mostram clustering de volatilidade
            - Teste ARCH rejeita H₀
            - Previsão de volatilidade é importante
            - Precificação de derivativos
            - Cálculo de VaR dinâmico
            """)
        
        with col2:
            st.markdown("""
            **❌ GARCH é overkill quando:**
            - Série é muito curta (< 100 obs)
            - Não há evidência de clustering
            - Teste ARCH não rejeita H₀
            - Volatilidade parece constante
            - Objetivo é apenas média condicional
            """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Sempre testa efeitos ARCH antes de modelar
    - Usa múltiplos pontos iniciais na estimação
    - Verifica se modelo estimado faz sentido econômico
    """)


def render_section_S5():
    """S5: Assimetria: quedas doem mais (GJR/EGARCH)"""
    st.header("↘️ Assimetria: Quedas Doem Mais")
    
    st.markdown("""
    GARCH simétrico trata choques positivos e negativos igualmente.
    Mas em finanças, **quedas aumentam mais a volatilidade** (efeito alavancagem).
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Modelos Assimétricos")
        
        st.markdown("""
        **GJR-GARCH:**
        $$h_t = \\omega + (\\alpha + \\gamma \\cdot I_{t-1}) r_{t-1}^2 + \\beta h_{t-1}$$
        
        Onde I = 1 se r < 0 (retorno negativo).
        
        **EGARCH (em log):**
        $$\\log h_t = \\omega + \\alpha |z_{t-1}| + \\gamma z_{t-1} + \\beta \\log h_{t-1}$$
        
        **Parâmetro γ (gamma):**
        - γ > 0: Retornos negativos aumentam mais a volatilidade
        - γ = 0: Simétrico (GARCH padrão)
        """)
        
        gamma = st.slider("γ (assimetria)", 0.0, 0.2, 0.1, 0.02, key="gamma_gjr")
        model = st.radio("Modelo:", ["GJR", "EGARCH"], horizontal=True, key="model_asym")
    
    with col2:
        # Simular
        data = simulate_asymmetric_garch(n=300, gamma=gamma, model=model, seed=42)
        
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=["Retornos", f"Volatilidade ({model})"],
                           row_heights=[0.5, 0.5])
        
        fig.add_trace(go.Scatter(y=data['returns'], mode='lines',
                                line=dict(width=0.8)), row=1, col=1)
        fig.add_trace(go.Scatter(y=data['volatility'], mode='lines',
                                line=dict(color='red')), row=2, col=1)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"asym_{gamma}_{model}")
    
    # News Impact Curve
    st.subheader("Curva de Impacto de Notícias")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **News Impact Curve:**
        > "Como a volatilidade de amanhã responde a choques de diferentes tamanhos?"
        
        - Eixo X: Tamanho do choque (r_{t-1})
        - Eixo Y: Variância futura (h_t)
        
        **Com assimetria:**
        - Curva é mais íngreme para choques negativos
        - Mesma magnitude, direções opostas → efeitos diferentes
        """)
    
    with col2:
        nic = news_impact_curve(omega=0.00001, alpha=0.1, beta=0.85, gamma=gamma)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nic['shocks'], y=nic['h_symmetric'],
                                mode='lines', name='GARCH Simétrico'))
        fig.add_trace(go.Scatter(x=nic['shocks'], y=nic['h_asymmetric'],
                                mode='lines', name=f'GJR (γ={gamma})'))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="News Impact Curve",
            xaxis_title="Choque (r_{t-1} / σ)",
            yaxis_title="Variância h_t",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True, key=f"nic_{gamma}")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa modelos assimétricos para risco de queda (downside risk)
    - GJR/EGARCH para VaR e stress testing
    - Importante para proteção de portfólios
    """)


def render_section_S6():
    """S6: Correlação dinâmica e aplicações estratégicas (DCC, hedge, beta)"""
    st.header("🔗 Correlação Dinâmica e Aplicações")
    
    st.markdown("""
    Correlações entre ativos **mudam no tempo**, especialmente em crises.
    Isso afeta hedge, diversificação e risco de portfólio.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 DCC", "🛡️ Hedge Dinâmico", "📈 Beta Variante"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Correlação Condicional Dinâmica")
            
            st.markdown("""
            **DCC-GARCH:**
            - Modela volatilidade de cada ativo (univariado)
            - Depois modela correlação que muda no tempo
            
            **Fato estilizado:**
            > "Correlações aumentam em crises"
            
            - Em períodos normais: correlação moderada
            - Em crises: correlação dispara → diversificação falha
            """)
            
            base_corr = st.slider("Correlação base", 0.2, 0.7, 0.5, 0.1, key="base_corr")
            crisis_corr = st.slider("Correlação na crise", 0.7, 0.99, 0.9, 0.05, key="crisis_corr")
        
        with col2:
            data = simulate_time_varying_corr(n=300, base_corr=base_corr, 
                                             crisis_corr=crisis_corr, seed=42)
            
            # Calcular DCC proxy
            dcc = compute_dcc_proxy(data['r1'], data['r2'], lambd=0.94)
            
            fig = make_subplots(rows=2, cols=1,
                               subplot_titles=["Retornos dos Ativos", "Correlação Condicional"],
                               row_heights=[0.5, 0.5])
            
            fig.add_trace(go.Scatter(y=data['r1'], name='Ativo 1',
                                    line=dict(width=0.8)), row=1, col=1)
            fig.add_trace(go.Scatter(y=data['r2'], name='Ativo 2',
                                    line=dict(width=0.8)), row=1, col=1)
            
            fig.add_trace(go.Scatter(y=data['true_corr'], name='Correlação Real',
                                    line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(y=dcc, name='DCC Estimada',
                                    line=dict(color='red', dash='dash')), row=2, col=1)
            
            # Marcar crise
            fig.add_vrect(x0=150, x1=200, fillcolor="yellow", opacity=0.2,
                         annotation_text="Crise", row=1, col=1)
            fig.add_vrect(x0=150, x1=200, fillcolor="yellow", opacity=0.2, row=2, col=1)
            
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True, key=f"dcc_{base_corr}_{crisis_corr}")
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Hedge Ratio Dinâmico")
            
            st.markdown("""
            **Hedge ratio ótimo:**
            $$h^* = \\frac{Cov(r_{asset}, r_{hedge})}{Var(r_{hedge})}$$
            
            **Problema:** Com variâncias e correlações que mudam, 
            o hedge ratio também deve mudar!
            
            **Exemplo:**
            - Hedge de ação com futuro de índice
            - Correlação aumenta em crise → hedge ratio muda
            - Não ajustar = sub ou sobre-hedging
            """)
        
        with col2:
            data = simulate_time_varying_corr(n=300, base_corr=0.6, 
                                             crisis_corr=0.9, seed=42)
            
            h_ratio = compute_dynamic_hedge_ratio(data['r1'], data['r2'], lambd=0.94)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=h_ratio, mode='lines', name='Hedge Ratio'))
            fig.add_hline(y=np.nanmean(h_ratio), line_dash="dash", line_color="red",
                         annotation_text=f"Média: {np.nanmean(h_ratio):.2f}")
            fig.add_vrect(x0=150, x1=200, fillcolor="yellow", opacity=0.2)
            
            fig.update_layout(
                title="Hedge Ratio Dinâmico",
                xaxis_title="Tempo",
                yaxis_title="h*",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("h* na calmaria", f"{np.nanmean(h_ratio[:140]):.2f}")
            col_m2.metric("h* na crise", f"{np.nanmean(h_ratio[150:200]):.2f}")
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Beta Variante no Tempo")
            
            st.markdown("""
            **CAPM assume beta constante:**
            $$r_i = \\alpha + \\beta r_m + \\varepsilon$$
            
            **Na realidade:**
            - Beta muda com condições de mercado
            - Em crises, muitos betas aumentam
            - Subestimar beta em crise = subestimar risco
            
            **Implicações:**
            - Alocação de risco incorreta
            - Capital mal dimensionado
            - Limites de risco inadequados
            """)
        
        with col2:
            data = simulate_time_varying_corr(n=300, base_corr=0.6, 
                                             crisis_corr=0.9, seed=42)
            
            beta_tv = compute_time_varying_beta(data['r1'], data['r2'], lambd=0.94)
            
            # Beta fixo (OLS)
            valid = ~np.isnan(beta_tv)
            beta_fix = np.cov(data['r1'][valid], data['r2'][valid])[0, 1] / np.var(data['r2'][valid])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=beta_tv, mode='lines', name='Beta Variante'))
            fig.add_hline(y=beta_fix, line_dash="dash", line_color="red",
                         annotation_text=f"Beta Fixo: {beta_fix:.2f}")
            fig.add_vrect(x0=150, x1=200, fillcolor="yellow", opacity=0.2)
            
            fig.update_layout(
                title="Beta: Fixo vs Variante no Tempo",
                xaxis_title="Tempo",
                yaxis_title="Beta",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Beta Fixo (OLS)", f"{beta_fix:.2f}")
            col_m2.metric("Beta na Crise", f"{np.nanmean(beta_tv[150:200]):.2f}")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Ajusta hedge ratio dinamicamente
    - Não confia em beta fixo para gestão de risco
    - Considera aumento de correlação em stress tests
    """)


def render_section_S7():
    """S7: Estudo de Caso MBA: VaR em crise (histórico vs EWMA vs GARCH)"""
    st.header("💼 Caso MBA: VaR em Crise")
    
    st.markdown("""
    Vamos comparar três métodos de VaR durante um período com choque de volatilidade.
    """)
    
    # Simular dados com crise
    np.random.seed(42)
    n = 300
    returns = np.zeros(n)
    
    # Período normal
    returns[:150] = np.random.normal(0, 1, 150)
    # Crise
    returns[150:200] = np.random.normal(0, 3, 50)
    # Recuperação
    returns[200:] = np.random.normal(0, 1.5, 100)
    
    # Calcular VaR pelos três métodos
    var_models = compute_var_models(returns, confidence=0.95)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Retornos e VaR")
        
        fig = go.Figure()
        
        # Retornos
        fig.add_trace(go.Scatter(y=returns, mode='lines', name='Retornos',
                                line=dict(width=0.8, color='blue')))
        
        # VaRs (negativos para comparar com perdas)
        fig.add_trace(go.Scatter(y=-var_models['var_hist'], mode='lines',
                                name='VaR Histórico', line=dict(color='green')))
        fig.add_trace(go.Scatter(y=-var_models['var_ewma'], mode='lines',
                                name='VaR EWMA', line=dict(color='orange')))
        fig.add_trace(go.Scatter(y=-var_models['var_garch'], mode='lines',
                                name='VaR GARCH', line=dict(color='red', dash='dash')))
        
        fig.add_vrect(x0=150, x1=200, fillcolor="red", opacity=0.1,
                     annotation_text="Crise")
        
        fig.update_layout(
            title="Retornos vs VaR 95% (3 métodos)",
            xaxis_title="Dias",
            yaxis_title="Retorno / -VaR (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Backtest: Violações")
        
        # Backtest
        bt_hist = backtest_var_exceedances(returns, var_models['var_hist'])
        bt_ewma = backtest_var_exceedances(returns, var_models['var_ewma'])
        bt_garch = backtest_var_exceedances(returns, var_models['var_garch'])
        
        results_df = pd.DataFrame({
            'Método': ['Histórico', 'EWMA', 'GARCH'],
            'Violações': [bt_hist['n_violations'], bt_ewma['n_violations'], bt_garch['n_violations']],
            'Taxa': [f"{bt_hist['violation_rate']*100:.1f}%", 
                    f"{bt_ewma['violation_rate']*100:.1f}%",
                    f"{bt_garch['violation_rate']*100:.1f}%"],
            'Esperado (5%)': ['5%', '5%', '5%']
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Interpretação:**
        - Taxa > 5%: Modelo subestima risco
        - Taxa < 5%: Modelo é conservador
        - Taxa ≈ 5%: Modelo bem calibrado
        """)
        
        # Violações no tempo
        fig2 = go.Figure()
        
        violations_hist = bt_hist['violations'].astype(int)
        violations_ewma = bt_ewma['violations'].astype(int)
        
        fig2.add_trace(go.Scatter(y=np.cumsum(violations_hist), mode='lines',
                                 name='Histórico'))
        fig2.add_trace(go.Scatter(y=np.cumsum(violations_ewma), mode='lines',
                                 name='EWMA'))
        
        # Linha esperada
        expected = np.arange(n) * 0.05
        fig2.add_trace(go.Scatter(y=expected, mode='lines', name='Esperado (5%)',
                                 line=dict(dash='dash', color='gray')))
        
        fig2.update_layout(
            title="Violações Acumuladas",
            xaxis_title="Dias",
            yaxis_title="# Violações",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("📋 Discussão")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Trade-offs:**
        - Histórico: simples, mas lento
        - EWMA: rápido, mas sem reversão
        - GARCH: completo, mas complexo
        """)
    
    with col2:
        st.markdown("""
        **Custos de Modelo:**
        - Histórico: quase zero
        - EWMA: baixo
        - GARCH: estimação, validação
        """)
    
    with col3:
        st.markdown("""
        **Governança:**
        - Documentar metodologia
        - Backtest regular
        - Comunicar limitações
        """)
    
    st.warning("""
    ⚠️ **Lição principal:** Em crises, todos os modelos tendem a falhar inicialmente.
    VaR dinâmico (EWMA/GARCH) reage mais rápido, mas ainda com atraso.
    Stress testing complementa VaR!
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Não depende de um único modelo de VaR
    - Faz backtest regularmente
    - Complementa VaR com stress testing
    - Comunica limitações aos stakeholders
    """)


def render_section_S8():
    """S8: Resumo Executivo e Ponte para o Próximo Módulo"""
    st.header("📋 Resumo Executivo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### O que Aprendemos sobre Volatilidade e Correlação
        
        ✅ **Fatos Estilizados:**
        - Volatility clustering: turbulência gera turbulência
        - Caudas pesadas: eventos extremos mais frequentes
        - Efeito alavancagem: quedas aumentam mais a volatilidade
        
        ✅ **Volatilidade Histórica vs EWMA:**
        - Histórica: simples, mas reação lenta
        - EWMA: mais rápido, mas sem reversão à média
        
        ✅ **GARCH(1,1):**
        - h_t = ω + α·r²_{t-1} + β·h_{t-1}
        - Persistência: α + β (próximo de 1 = memória longa)
        - Previsão de volatilidade para VaR e opções
        
        ✅ **Estimação e Diagnóstico:**
        - Máxima verossimilhança (cuidado com ótimos locais)
        - Teste ARCH-LM antes de modelar
        
        ✅ **Assimetria (GJR/EGARCH):**
        - Retornos negativos aumentam mais a volatilidade
        - News Impact Curve mostra assimetria
        
        ✅ **Correlação Dinâmica:**
        - DCC: correlação aumenta em crises
        - Hedge ratio e beta variam no tempo
        - Diversificação falha quando mais precisamos
        
        ✅ **VaR:**
        - Comparar métodos: histórico, EWMA, GARCH
        - Backtest: verificar violações
        - Complementar com stress testing
        """)
    
    with col2:
        st.markdown("### 💡 Mensagem-Chave")
        
        st.info("""
        **"Risco muda com o tempo"**
        
        Modelos com variância constante falham em finanças.
        
        GARCH e DCC capturam dinâmica de risco essencial para:
        - VaR e limites de risco
        - Hedge e alocação
        - Precificação de derivativos
        """)
        
        st.markdown("### 🧪 Quiz Final")
        
        resposta = st.radio(
            "Se α + β = 0.98, o que isso significa?",
            ["Volatilidade é praticamente constante",
             "Choques se dissipam em poucos dias",
             "Choques têm efeito muito persistente"],
            key="quiz_final"
        )
        
        if st.button("Ver resposta", key="btn_final"):
            if resposta == "Choques têm efeito muito persistente":
                st.success("""
                ✅ **Correto!**
                
                Persistência de 0.98 significa:
                - Meia-vida ≈ 34 dias
                - Choques demoram muito para dissipar
                - Volatilidade alta persiste por semanas
                """)
            else:
                st.error("Alta persistência (próximo de 1) = memória longa, choques demoram a dissipar.")
    
    st.markdown("---")
    
    st.subheader("🔜 Próximo Módulo: Dados em Painel")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Estrutura de Painel:**
        - Múltiplas unidades (empresas, países)
        - Múltiplos períodos
        - Heterogeneidade
        """)
    
    with col2:
        st.markdown("""
        **Modelos:**
        - Efeitos fixos
        - Efeitos aleatórios
        - GMM dinâmico
        """)
    
    with col3:
        st.markdown("""
        **Aplicações:**
        - Finanças corporativas
        - Macroeconomia
        - Organização industrial
        """)
    
    st.success("""
    🎓 **Mensagem final:** Volatilidade e correlação não são constantes.
    Modelos dinâmicos (GARCH, DCC) são essenciais para gestão de risco moderna.
    Combine com backtest e stress testing para decisões robustas.
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Implementa modelos de volatilidade condicional para VaR
    - Monitora correlações em tempo real
    - Ajusta hedge e limites de risco dinamicamente
    - Não esquece: modelos falham em crises extremas
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("📊 Módulo 9: Modelagem de Volatilidade e Correlação")
    st.markdown("**Laboratório de Econometria** | GARCH, Assimetria e DCC")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Explicar **fatos estilizados** de retornos financeiros
        - Comparar **volatilidade histórica** e **EWMA**
        - Introduzir **GARCH(1,1)** como modelo de risco variável
        - Ensinar **estimação** por máxima verossimilhança e **diagnóstico**
        - Mostrar **assimetria** (GJR/EGARCH) e impacto de notícias
        - Introduzir **correlação dinâmica** (DCC) e aplicações
        - Comparar métodos de **VaR** e fazer backtest
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "📊 Fatos Estilizados",
        "S2": "📏 Histórica vs EWMA",
        "S3": "📈 GARCH(1,1)",
        "S4": "🔧 Estimação e Diagnóstico",
        "S5": "↘️ Assimetria",
        "S6": "🔗 Correlação Dinâmica",
        "S7": "💼 Caso: VaR",
        "S8": "📋 Resumo"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Dica:** Volatilidade que muda 
    no tempo é a base da gestão 
    de risco moderna em finanças.
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


# =============================================================================
# EXECUÇÃO STANDALONE (para testes)
# =============================================================================

if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Módulo 9: Volatilidade e Correlação",
            page_icon="📊",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()