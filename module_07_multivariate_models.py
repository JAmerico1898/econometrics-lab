"""
Laboratório de Econometria - Module 7: Multivariate Models
Aplicativo educacional interativo para modelos multivariados (SEM, IV/2SLS, VAR).
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

def simulate_simultaneous_system(n: int = 200, simultaneity: float = 0.5, seed: int = 42) -> dict:
    """
    Simula sistema de oferta-demanda com simultaneidade.
    Preço e quantidade se determinam mutuamente.
    """
    np.random.seed(seed)
    
    # Exógenas
    renda = np.random.normal(100, 20, n)  # Afeta demanda
    custo = np.random.normal(50, 10, n)   # Afeta oferta
    
    # Erros estruturais
    e_d = np.random.normal(0, 5, n)  # Erro da demanda
    e_s = np.random.normal(0, 5, n)  # Erro da oferta
    
    # Sistema estrutural (forma reduzida para resolver):
    # Demanda: Q = a0 + a1*P + a2*Renda + e_d
    # Oferta:  Q = b0 + b1*P + b2*Custo + e_s
    # Parâmetros verdadeiros
    a0, a1, a2 = 50, -0.5, 0.3   # Demanda: P aumenta -> Q diminui
    b0, b1, b2 = 10, 0.8, -0.2   # Oferta: P aumenta -> Q aumenta
    
    # Resolver para P e Q (forma reduzida)
    # P = (a0 - b0 + a2*Renda - b2*Custo + e_d - e_s) / (b1 - a1)
    # Q = a0 + a1*P + a2*Renda + e_d
    
    denom = b1 - a1  # = 0.8 - (-0.5) = 1.3
    P = (a0 - b0 + a2*renda - b2*custo + simultaneity*(e_d - e_s)) / denom
    Q = a0 + a1*P + a2*renda + e_d
    
    return {
        'P': P,
        'Q': Q,
        'Renda': renda,
        'Custo': custo,
        'true_a1': a1,  # Coeficiente verdadeiro de P na demanda
        'true_b1': b1   # Coeficiente verdadeiro de P na oferta
    }


def fit_ols_simple(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS simples retornando coeficientes e estatísticas."""
    n, k = X.shape
    
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
        'y_hat': y_hat
    }


def fit_2sls(y: np.ndarray, X_endog: np.ndarray, X_exog: np.ndarray, Z: np.ndarray) -> dict:
    """
    Estimação 2SLS (Two-Stage Least Squares).
    y: variável dependente
    X_endog: variáveis endógenas
    X_exog: variáveis exógenas (incluídas)
    Z: instrumentos (excluídos)
    """
    n = len(y)
    
    # Primeiro estágio: regredir X_endog em [X_exog, Z]
    W = np.column_stack([X_exog, Z])
    first_stage = fit_ols_simple(X_endog, W)
    X_endog_hat = first_stage['y_hat']
    
    # Segundo estágio: regredir y em [X_exog, X_endog_hat]
    X_second = np.column_stack([X_exog, X_endog_hat])
    second_stage = fit_ols_simple(y, X_second)
    
    # F-stat do primeiro estágio (força do instrumento)
    # Simplificado: R² do primeiro estágio
    f_stat_first = (first_stage['r_squared'] / (1 - first_stage['r_squared'])) * (n - W.shape[1]) / (Z.shape[1] if Z.ndim > 1 else 1)
    
    return {
        'beta': second_stage['beta'],
        'se': second_stage['se'],
        'r_squared': second_stage['r_squared'],
        'first_stage_r2': first_stage['r_squared'],
        'first_stage_f': f_stat_first,
        'X_endog_hat': X_endog_hat
    }


def simulate_iv_scenario(n: int = 300, instrument_valid: bool = True, seed: int = 42) -> dict:
    """Simula cenário com instrumento válido ou inválido."""
    np.random.seed(seed)
    
    # Confundidor (não observado)
    U = np.random.normal(0, 1, n)
    
    # Instrumento Z
    Z = np.random.normal(0, 1, n)
    
    # X endógeno: afetado por Z e pelo confundidor U
    X = 2 + 0.8 * Z + 0.6 * U + np.random.normal(0, 0.5, n)
    
    # Y: efeito verdadeiro de X é 1.5
    # Se instrumento inválido, Z também afeta Y diretamente
    if instrument_valid:
        Y = 1 + 1.5 * X + 0.7 * U + np.random.normal(0, 1, n)
    else:
        Y = 1 + 1.5 * X + 0.7 * U + 0.5 * Z + np.random.normal(0, 1, n)  # Z afeta Y diretamente!
    
    return {
        'Y': Y,
        'X': X,
        'Z': Z,
        'U': U,
        'true_effect': 1.5
    }


def hausman_test_simple(beta_ols: float, beta_iv: float, se_ols: float, se_iv: float) -> dict:
    """Teste de Hausman simplificado para endogeneidade."""
    # H = (beta_IV - beta_OLS)^2 / (Var(beta_IV) - Var(beta_OLS))
    diff = beta_iv - beta_ols
    var_diff = se_iv**2 - se_ols**2
    
    if var_diff <= 0:
        # Usar aproximação quando variância é negativa
        var_diff = se_iv**2
    
    h_stat = diff**2 / var_diff
    p_value = 1 - stats.chi2.cdf(h_stat, 1)
    
    return {
        'h_stat': h_stat,
        'p_value': p_value,
        'diff': diff
    }


def simulate_var_data(n: int = 200, a12: float = 0.3, a21: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """
    Simula VAR(1) bivariado.
    y1_t = c1 + a11*y1_{t-1} + a12*y2_{t-1} + e1_t
    y2_t = c2 + a21*y1_{t-1} + a22*y2_{t-1} + e2_t
    """
    np.random.seed(seed)
    
    # Parâmetros
    c1, c2 = 0.5, 0.3
    a11, a22 = 0.5, 0.4  # Persistência própria
    
    # Inicializar
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    
    e1 = np.random.normal(0, 1, n)
    e2 = np.random.normal(0, 1, n)
    
    for t in range(1, n):
        y1[t] = c1 + a11*y1[t-1] + a12*y2[t-1] + e1[t]
        y2[t] = c2 + a21*y1[t-1] + a22*y2[t-1] + e2[t]
    
    return pd.DataFrame({
        'y1': y1,
        'y2': y2,
        't': np.arange(n)
    })


def fit_var1_simple(y1: np.ndarray, y2: np.ndarray) -> dict:
    """Ajusta VAR(1) bivariado simples."""
    n = len(y1)
    
    # Lags
    y1_lag = y1[:-1]
    y2_lag = y2[:-1]
    y1_curr = y1[1:]
    y2_curr = y2[1:]
    
    # Equação 1: y1_t = c1 + a11*y1_{t-1} + a12*y2_{t-1}
    X = np.column_stack([np.ones(n-1), y1_lag, y2_lag])
    eq1 = fit_ols_simple(y1_curr, X)
    
    # Equação 2: y2_t = c2 + a21*y1_{t-1} + a22*y2_{t-1}
    eq2 = fit_ols_simple(y2_curr, X)
    
    # Matriz de coeficientes A
    A = np.array([
        [eq1['beta'][1], eq1['beta'][2]],
        [eq2['beta'][1], eq2['beta'][2]]
    ])
    
    return {
        'eq1_beta': eq1['beta'],
        'eq2_beta': eq2['beta'],
        'A': A,
        'eq1_residuals': eq1['residuals'],
        'eq2_residuals': eq2['residuals']
    }


def granger_test_simple(y1: np.ndarray, y2: np.ndarray, max_lag: int = 4) -> dict:
    """Teste de Granger simplificado: y2 Granger-causa y1?"""
    n = len(y1)
    
    # Modelo restrito: y1_t = c + a*y1_{t-1} (sem y2)
    y1_lag = y1[1:-1]
    y1_curr = y1[2:]
    X_r = np.column_stack([np.ones(len(y1_curr)), y1_lag])
    ols_r = fit_ols_simple(y1_curr, X_r)
    sse_r = np.sum(ols_r['residuals']**2)
    
    # Modelo irrestrito: y1_t = c + a*y1_{t-1} + b*y2_{t-1}
    y2_lag = y2[1:-1]
    X_ur = np.column_stack([np.ones(len(y1_curr)), y1_lag, y2_lag])
    ols_ur = fit_ols_simple(y1_curr, X_ur)
    sse_ur = np.sum(ols_ur['residuals']**2)
    
    # Teste F
    q = 1  # Uma restrição (coeficiente de y2_lag = 0)
    k = X_ur.shape[1]
    n_obs = len(y1_curr)
    
    f_stat = ((sse_r - sse_ur) / q) / (sse_ur / (n_obs - k))
    p_value = 1 - stats.f.cdf(f_stat, q, n_obs - k)
    
    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'sse_r': sse_r,
        'sse_ur': sse_ur
    }


def compute_irf(A: np.ndarray, periods: int = 20, shock_var: int = 0) -> np.ndarray:
    """
    Computa Impulse Response Function para VAR(1).
    A: matriz de coeficientes 2x2
    shock_var: índice da variável que recebe o choque (0 ou 1)
    """
    k = A.shape[0]
    irf = np.zeros((periods, k))
    
    # Choque inicial
    shock = np.zeros(k)
    shock[shock_var] = 1.0
    
    irf[0] = shock
    
    # Propagar
    for t in range(1, periods):
        irf[t] = A @ irf[t-1]
    
    return irf


def compute_fevd(A: np.ndarray, periods: int = 20) -> np.ndarray:
    """
    Computa Forecast Error Variance Decomposition simplificada.
    Retorna proporção da variância de cada variável explicada por choques próprios vs externos.
    """
    k = A.shape[0]
    fevd = np.zeros((periods, k, k))  # [período, variável, fonte do choque]
    
    # IRF para cada choque
    irfs = [compute_irf(A, periods, i) for i in range(k)]
    
    for h in range(periods):
        for i in range(k):
            total_var = sum(irfs[j][:h+1, i]**2 for j in range(k))
            total_var = np.sum(total_var)
            if total_var > 0:
                for j in range(k):
                    fevd[h, i, j] = np.sum(irfs[j][:h+1, i]**2) / total_var
    
    return fevd


def make_real_estate_case_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Gera dados sintéticos de retornos imobiliários e variáveis macro."""
    np.random.seed(seed)
    
    # Simular VAR(1) com 3 variáveis
    juros = np.zeros(n)
    inflacao = np.zeros(n)
    retorno_imob = np.zeros(n)
    
    e1 = np.random.normal(0, 0.5, n)
    e2 = np.random.normal(0, 0.3, n)
    e3 = np.random.normal(0, 1.5, n)
    
    juros[0] = 5
    inflacao[0] = 3
    retorno_imob[0] = 8
    
    for t in range(1, n):
        juros[t] = 1 + 0.7*juros[t-1] + 0.2*inflacao[t-1] + e1[t]
        inflacao[t] = 0.5 + 0.1*juros[t-1] + 0.6*inflacao[t-1] + e2[t]
        retorno_imob[t] = 2 - 0.5*juros[t-1] + 0.3*inflacao[t-1] + 0.4*retorno_imob[t-1] + e3[t]
    
    return pd.DataFrame({
        'Juros': juros,
        'Inflacao': inflacao,
        'Retorno_Imob': retorno_imob,
        't': np.arange(n)
    })


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: Introdução: Por que Modelos Multivariados?"""
    st.header("🔄 Por que Modelos Multivariados?")
    
    st.markdown("""
    Em muitos problemas de negócio, **variáveis se influenciam mutuamente**.
    Regressões de equação única falham quando há essa interdependência.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("O Problema")
        
        st.markdown("""
        **Pergunta de negócio:**
        > "Como marketing afeta vendas se o orçamento de marketing 
        > depende das vendas passadas?"
        
        **Equação única (OLS):**
        - Assume que X afeta Y, mas Y não afeta X
        - Ignora feedback e interdependência
        - Coeficientes ficam **enviesados**
        
        **Abordagem de sistema:**
        - Modela múltiplas variáveis simultaneamente
        - Captura feedback entre variáveis
        - Permite identificar efeitos causais
        """)
    
    with col2:
        st.subheader("Exemplo: Mercado Imobiliário")
        
        st.markdown("""
        **Preço e Quantidade se determinam juntos:**
        
        ```
        Demanda: Q = f(P, Renda, ...)
                 ↓
                 P afeta Q
        
        Oferta:  Q = g(P, Custos, ...)
                 ↓
                 P afeta Q
        
        Mas P é determinado pelo encontro de oferta e demanda!
        ```
        
        **Feedback mútuo:**
        - Alta demanda → Preços sobem
        - Preços altos → Oferta aumenta
        - Mais oferta → Preços caem
        - E o ciclo continua...
        """)
        
        st.warning("""
        ⚠️ Se você estimar apenas a equação de demanda com OLS,
        o coeficiente de P estará **enviesado** porque P é endógeno!
        """)
    
    # Visual do sistema
    st.subheader("Equação Única vs Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Equação Única:**
        ```
        Y = α + β·X + ε
        
        Assume: X → Y (só uma direção)
        ```
        """)
    
    with col2:
        st.markdown("""
        **Sistema de Equações:**
        ```
        Y₁ = α₁ + β₁·Y₂ + γ₁·X₁ + ε₁
        Y₂ = α₂ + β₂·Y₁ + γ₂·X₂ + ε₂
        
        Captura: Y₁ ↔ Y₂ (feedback)
        ```
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Reconhece quando variáveis se influenciam mutuamente
    - Usa modelos de sistema (SEM, VAR) em vez de regressão simples
    """)


def render_section_S2():
    """S2: Equações Simultâneas (SEM): Endogeneidade e Viés de Simultaneidade"""
    st.header("⚡ Viés de Simultaneidade")
    
    st.markdown("""
    Quando variáveis se determinam simultaneamente, OLS em uma única equação 
    produz **coeficientes enviesados**. Isso é o **viés de simultaneidade**.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Sistema Oferta-Demanda")
        
        st.markdown("""
        **Demanda:** Q = α₀ + α₁·P + α₂·Renda + ε_d
        
        **Oferta:** Q = β₀ + β₁·P + β₂·Custo + ε_s
        
        **Parâmetros verdadeiros:**
        - α₁ = -0.5 (preço ↑ → demanda ↓)
        - β₁ = +0.8 (preço ↑ → oferta ↑)
        """)
        
        simultaneity = st.slider(
            "Intensidade da simultaneidade",
            0.0, 1.0, 0.5, 0.1,
            key="simult_slider",
            help="0 = sem simultaneidade; 1 = simultaneidade total"
        )
        
        st.markdown("""
        **Endógenas vs Exógenas:**
        - **Endógenas:** P e Q (determinadas pelo sistema)
        - **Exógenas:** Renda, Custo (determinadas fora do sistema)
        """)
    
    with col2:
        # Simular sistema
        data = simulate_simultaneous_system(n=300, simultaneity=simultaneity, seed=42)
        
        # OLS na equação de demanda (enviesado)
        X_ols = np.column_stack([np.ones(300), data['P'], data['Renda']])
        ols_result = fit_ols_simple(data['Q'], X_ols)
        
        beta_p_ols = ols_result['beta'][1]
        true_beta = data['true_a1']
        vies = beta_p_ols - true_beta
        
        st.subheader("Resultado: OLS na Demanda")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("β verdadeiro (P)", f"{true_beta:.2f}")
        col_m2.metric("β OLS", f"{beta_p_ols:.2f}")
        col_m3.metric("Viés", f"{vies:.2f}", delta_color="inverse")
        
        # Gráfico
        fig = px.scatter(x=data['P'], y=data['Q'], opacity=0.5,
                        labels={'x': 'Preço', 'y': 'Quantidade'},
                        title="Preço vs Quantidade (Dados Simultâneos)")
        
        # Linha OLS
        p_range = np.linspace(data['P'].min(), data['P'].max(), 50)
        q_ols = ols_result['beta'][0] + beta_p_ols * p_range + ols_result['beta'][2] * np.mean(data['Renda'])
        fig.add_trace(go.Scatter(x=p_range, y=q_ols, mode='lines',
                                name=f'OLS: β={beta_p_ols:.2f}', line=dict(color='red')))
        
        # Linha verdadeira
        q_true = 50 + true_beta * p_range + 0.3 * np.mean(data['Renda'])
        fig.add_trace(go.Scatter(x=p_range, y=q_true, mode='lines',
                                name=f'Verdadeiro: β={true_beta:.2f}', 
                                line=dict(color='green', dash='dash')))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key=f"fig_simult_{simultaneity}")
    
    if abs(vies) > 0.1:
        st.error(f"""
        🚨 **Viés significativo!** OLS estima β = {beta_p_ols:.2f}, 
        mas o verdadeiro é {true_beta:.2f}. Diferença de {abs(vies/true_beta)*100:.0f}%!
        """)
    else:
        st.success("✅ Com baixa simultaneidade, o viés é pequeno.")
    
    with st.expander("📖 Por que OLS falha?"):
        st.markdown("""
        **O problema técnico:**
        
        Em OLS, assumimos que Cov(X, ε) = 0 (exogeneidade).
        
        Mas em sistemas simultâneos:
        - P é determinado junto com Q
        - Choques na demanda (ε_d) afetam P via equilíbrio
        - Logo, Cov(P, ε_d) ≠ 0 — **P é endógeno**
        
        **Resultado:** OLS não consegue separar o efeito de P sobre Q do efeito de Q sobre P.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Desconfia de estimativas OLS quando há feedback entre variáveis
    - Busca métodos que lidem com endogeneidade (IV, 2SLS)
    """)


def render_section_S3():
    """S3: Forma Estrutural vs Forma Reduzida"""
    st.header("📐 Forma Estrutural vs Forma Reduzida")
    
    st.markdown("""
    A **forma estrutural** representa a teoria econômica. 
    A **forma reduzida** é o que conseguimos estimar diretamente.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Forma Estrutural (Teoria)")
        
        st.markdown("""
        **Equações que refletem comportamento:**
        
        **Demanda:**
        $$Q^d = \\alpha_0 + \\alpha_1 P + \\alpha_2 Renda + \\varepsilon_d$$
        
        **Oferta:**
        $$Q^s = \\beta_0 + \\beta_1 P + \\beta_2 Custo + \\varepsilon_s$$
        
        **Equilíbrio:** Q^d = Q^s = Q
        
        **Problema:** P aparece do lado direito, mas é endógeno!
        Não dá para estimar diretamente com OLS.
        """)
    
    with col2:
        st.subheader("Forma Reduzida (Estimável)")
        
        st.markdown("""
        **Resolver o sistema para P e Q:**
        
        Substituindo e resolvendo:
        
        $$P = \\pi_0 + \\pi_1 Renda + \\pi_2 Custo + v_P$$
        
        $$Q = \\gamma_0 + \\gamma_1 Renda + \\gamma_2 Custo + v_Q$$
        
        **Agora:** P e Q dependem apenas de **exógenas** (Renda, Custo).
        
        **Podemos estimar com OLS!**
        """)
    
    st.markdown("---")
    
    # Diagrama visual
    st.subheader("Diagrama do Sistema")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────┐
        │                    SISTEMA                       │
        │                                                  │
        │   Renda ──────┐                ┌────── Custo    │
        │               │                │                 │
        │               ▼                ▼                 │
        │           ┌───────┐        ┌───────┐            │
        │           │Demanda│◄──────►│Oferta │            │
        │           └───┬───┘        └───┬───┘            │
        │               │                │                 │
        │               └───────┬────────┘                │
        │                       │                          │
        │                       ▼                          │
        │                   ┌───────┐                      │
        │                   │ P, Q  │  ← Endógenas        │
        │                   └───────┘                      │
        │                                                  │
        │   Renda, Custo = Exógenas (pré-determinadas)    │
        └─────────────────────────────────────────────────┘
        ```
        """)
    
    # Toggle para ver parâmetros
    with st.expander("🔢 Ver como parâmetros se relacionam"):
        st.markdown("""
        **Da estrutural para a reduzida:**
        
        Os π's e γ's da forma reduzida são **combinações** dos parâmetros estruturais:
        
        | Parâmetro Reduzido | Fórmula |
        |-------------------|---------|
        | π₁ (efeito de Renda em P) | α₂ / (β₁ - α₁) |
        | π₂ (efeito de Custo em P) | -β₂ / (β₁ - α₁) |
        | γ₁ (efeito de Renda em Q) | α₂β₁ / (β₁ - α₁) |
        | γ₂ (efeito de Custo em Q) | -α₁β₂ / (β₁ - α₁) |
        
        **O desafio:** Temos 4 parâmetros reduzidos, mas 6 estruturais.
        Precisamos de **restrições de identificação** para recuperar os estruturais.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Entende que a forma reduzida prevê bem, mas não explica mecanismos
    - Para entender causalidade, precisa identificar a forma estrutural
    """)


def render_section_S4():
    """S4: Identificação: Quando é Possível Recuperar a Teoria?"""
    st.header("🔍 Identificação: Recuperando Parâmetros Estruturais")
    
    st.markdown("""
    **Identificação** é a possibilidade de recuperar os parâmetros teóricos (estruturais)
    a partir dos dados. Nem sempre é possível!
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Condição de Ordem")
        
        st.markdown("""
        **Regra prática:** Para identificar uma equação, você precisa de 
        variáveis **excluídas** dessa equação mas presentes em outras.
        
        **Condição de Ordem:**
        > Número de variáveis excluídas ≥ Número de endógenas - 1
        
        **Exemplo (Demanda):**
        - Endógenas: P, Q (2 variáveis)
        - Precisa: ≥ 1 variável excluída da demanda
        - Custo está na oferta, não na demanda ✓
        
        **Status:**
        - **Sub-identificada:** Não há exclusões suficientes
        - **Exatamente identificada:** Número exato de exclusões
        - **Sobre-identificada:** Mais exclusões que o necessário
        """)
    
    with col2:
        st.subheader("Exemplo Visual")
        
        st.markdown("""
        | Equação | Variáveis | Excluídas | Status |
        |---------|-----------|-----------|--------|
        | Demanda | P, Q, Renda | Custo | ✅ Identificada |
        | Oferta | P, Q, Custo | Renda | ✅ Identificada |
        
        **Por que funciona?**
        - Custo afeta oferta mas NÃO demanda diretamente
        - Renda afeta demanda mas NÃO oferta diretamente
        - Essas exclusões permitem separar as equações
        """)
        
        st.info("""
        💡 **Intuição:** Cada exclusão é uma "alavanca" que move uma equação
        sem mover a outra, permitindo identificar o efeito.
        """)
    
    # Quiz
    st.subheader("🧪 Quiz: Esta Equação Está Identificada?")
    
    st.markdown("""
    **Sistema:**
    - Equação 1: Y₁ = α + β·Y₂ + γ·X₁ + ε₁
    - Equação 2: Y₂ = δ + θ·Y₁ + λ·X₁ + μ·X₂ + ε₂
    
    **Pergunta:** A Equação 1 está identificada?
    """)
    
    resposta = st.radio(
        "Selecione:",
        ["Não identificada", "Exatamente identificada", "Sobre-identificada"],
        key="quiz_ident"
    )
    
    if st.button("Ver resposta", key="btn_ident"):
        if resposta == "Exatamente identificada":
            st.success("""
            ✅ **Correto!**
            
            - Endógenas: Y₁, Y₂ (2 variáveis)
            - Condição: precisa ≥ 1 exclusão
            - X₂ aparece na Eq. 2 mas NÃO na Eq. 1
            - Temos 1 exclusão = exatamente o necessário
            - Equação 1 é **exatamente identificada**
            """)
        else:
            st.error("""
            A Eq. 1 é **exatamente identificada**. 
            X₂ é excluída da Eq. 1 mas presente na Eq. 2, 
            fornecendo a exclusão necessária.
            """)
    
    with st.expander("📖 Teste de Hausman (Endogeneidade)"):
        st.markdown("""
        **Intuição do Teste de Hausman:**
        
        Compara estimativas OLS e IV:
        - Se X é exógeno: OLS e IV devem dar resultados similares
        - Se X é endógeno: OLS é viesado, IV não — resultados diferentes
        
        **Hipóteses:**
        - H₀: X é exógeno (OLS é consistente)
        - H₁: X é endógeno (precisamos de IV)
        
        **Estatística:** H = (β_IV - β_OLS)² / [Var(β_IV) - Var(β_OLS)]
        
        Se H é grande (p < 0.05), rejeita H₀ → **X é endógeno, use IV!**
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Verifica se há exclusões válidas antes de estimar
    - Usa teste de Hausman para confirmar necessidade de IV
    """)


def render_section_S5():
    """S5: Estimação: IV e 2SLS (a solução padrão)"""
    st.header("🔧 Variáveis Instrumentais e 2SLS")
    
    st.markdown("""
    **Variáveis Instrumentais (IV)** resolvem o problema de endogeneidade usando 
    uma variável que afeta X mas não afeta Y diretamente.
    """)
    
    tab1, tab2, tab3 = st.tabs(["💡 Intuição", "🔬 Simulação", "📊 2SLS Passo-a-Passo"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("O que é um Instrumento?")
            
            st.markdown("""
            **Instrumento Z é válido se:**
            
            1. **Relevância:** Z afeta X (correlacionado com X)
            2. **Exogeneidade:** Z não afeta Y diretamente (só via X)
            
            **Diagrama:**
            ```
                    U (confundidor)
                   ↙ ↘
            Z → X    →    Y
                ↖─────────┘
                (Z não pode ter seta direta para Y!)
            ```
            
            **Intuição:**
            - Z "empurra" X de forma exógena
            - Usamos só a variação em X que veio de Z
            - Essa variação "limpa" não está contaminada por U
            """)
        
        with col2:
            st.subheader("Exemplos de Instrumentos")
            
            st.markdown("""
            | Problema | Endógena | Instrumento |
            |----------|----------|-------------|
            | Educação → Salário | Educação | Proximidade de universidade |
            | Preço → Demanda | Preço | Custo de produção |
            | Publicidade → Vendas | Publicidade | Preço de mídia |
            | Crédito → Consumo | Crédito | Regulação bancária |
            
            **O desafio:** Encontrar instrumentos válidos é DIFÍCIL!
            """)
            
            st.warning("""
            ⚠️ **Cuidado:** Se Z afeta Y diretamente (além de via X),
            o instrumento é **inválido** e IV ainda será viesado!
            """)
    
    with tab2:
        st.subheader("Simulação: Instrumento Válido vs Inválido")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            instrumento_valido = st.radio(
                "Tipo de instrumento:",
                ["✅ Válido (Z não afeta Y diretamente)",
                 "❌ Inválido (Z afeta Y diretamente)"],
                key="iv_tipo"
            )
            
            is_valid = "Válido" in instrumento_valido
            
            st.markdown(f"""
            **Efeito verdadeiro de X sobre Y:** 1.50
            
            **Cenário:** {"Z é um bom instrumento" if is_valid else "Z afeta Y diretamente (violação!)"}
            """)
        
        with col2:
            # Simular
            data = simulate_iv_scenario(n=500, instrument_valid=is_valid, seed=42)
            
            # OLS (viesado)
            X_ols = np.column_stack([np.ones(500), data['X']])
            ols = fit_ols_simple(data['Y'], X_ols)
            
            # 2SLS
            X_exog = np.ones((500, 1))
            iv = fit_2sls(data['Y'], data['X'], X_exog, data['Z'].reshape(-1, 1))
            
            true_effect = data['true_effect']
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("β Verdadeiro", f"{true_effect:.2f}")
            col_m2.metric("β OLS", f"{ols['beta'][1]:.2f}",
                         delta=f"Viés: {ols['beta'][1] - true_effect:.2f}")
            col_m3.metric("β IV/2SLS", f"{iv['beta'][1]:.2f}",
                         delta=f"Viés: {iv['beta'][1] - true_effect:.2f}")
            
            if is_valid:
                st.success("✅ IV corrige o viés! Estimativa próxima do verdadeiro.")
            else:
                st.error("❌ Com instrumento inválido, IV também é viesado!")
    
    with tab3:
        st.subheader("2SLS: Dois Estágios")
        
        st.markdown("""
        **2SLS (Two-Stage Least Squares)** implementa IV em dois passos:
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **1º Estágio: Limpar X**
            
            Regredir X endógeno contra instrumentos Z:
            $$X = \\gamma_0 + \\gamma_1 Z + v$$
            
            Obter $\\hat{X}$ (parte de X explicada por Z)
            
            **2º Estágio: Estimar efeito**
            
            Regredir Y contra $\\hat{X}$:
            $$Y = \\alpha + \\beta \\hat{X} + \\varepsilon$$
            
            β é o efeito causal "limpo"!
            """)
        
        with col2:
            # Visualizar os dois estágios
            data = simulate_iv_scenario(n=300, instrument_valid=True, seed=123)
            
            # Primeiro estágio
            X_first = np.column_stack([np.ones(300), data['Z']])
            first = fit_ols_simple(data['X'], X_first)
            X_hat = first['y_hat']
            
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=["1º Estágio: X vs Z", "2º Estágio: Y vs X̂"])
            
            fig.add_trace(go.Scatter(x=data['Z'], y=data['X'], mode='markers',
                                    opacity=0.5, name='Dados'), row=1, col=1)
            z_range = np.linspace(data['Z'].min(), data['Z'].max(), 50)
            fig.add_trace(go.Scatter(x=z_range, y=first['beta'][0] + first['beta'][1]*z_range,
                                    mode='lines', name='X̂ = f(Z)', line=dict(color='red')),
                         row=1, col=1)
            
            fig.add_trace(go.Scatter(x=X_hat, y=data['Y'], mode='markers',
                                    opacity=0.5, name='Dados'), row=1, col=2)
            
            fig.update_layout(height=350, showlegend=False)
            fig.update_xaxes(title_text="Z (instrumento)", row=1, col=1)
            fig.update_xaxes(title_text="X̂ (valor previsto)", row=1, col=2)
            fig.update_yaxes(title_text="X", row=1, col=1)
            fig.update_yaxes(title_text="Y", row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("R² do 1º Estágio", f"{first['r_squared']:.3f}",
                     help="Quanto Z explica X. Deve ser razoavelmente alto!")
    
    with st.expander("⚖️ Trade-off: Viés vs Variância"):
        st.markdown("""
        **OLS:**
        - Viesado (se X endógeno)
        - Baixa variância (usa toda informação)
        
        **IV/2SLS:**
        - Não viesado (se instrumento válido)
        - Alta variância (usa só parte da informação)
        
        **Na prática:**
        - Se endogeneidade é forte → Use IV
        - Se instrumento é fraco (R² baixo no 1º estágio) → IV pode ter variância enorme
        - Regra: F-stat do 1º estágio > 10
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Busca instrumentos válidos quando suspeita de endogeneidade
    - Verifica força do instrumento (R² do 1º estágio)
    - Aceita mais incerteza em troca de menos viés
    """)


def render_section_S6():
    """S6: VAR: Modelagem Multivariada em Séries Temporais"""
    st.header("📈 VAR: Vetores Autoregressivos")
    
    st.markdown("""
    **VAR** trata todas as variáveis como potencialmente endógenas e 
    modela a dinâmica conjunta usando seus próprios lags.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("O que é VAR?")
        
        st.markdown("""
        **VAR(1) com 2 variáveis:**
        
        $$y_{1,t} = c_1 + a_{11} y_{1,t-1} + a_{12} y_{2,t-1} + \\varepsilon_{1,t}$$
        $$y_{2,t} = c_2 + a_{21} y_{1,t-1} + a_{22} y_{2,t-1} + \\varepsilon_{2,t}$$
        
        **Características:**
        - Cada variável depende de seus lags E dos lags das outras
        - Não precisa especificar quem causa quem a priori
        - Captura dinâmica conjunta
        """)
        
        st.subheader("Vantagens e Desvantagens")
        
        col_v, col_d = st.columns(2)
        with col_v:
            st.markdown("""
            **✅ Vantagens:**
            - Flexível, a-teórico
            - Bom para previsão
            - Ferramentas ricas (IRF, FEVD)
            """)
        with col_d:
            st.markdown("""
            **❌ Desvantagens:**
            - Muitos parâmetros
            - Difícil interpretar
            - Sensível à ordenação
            """)
    
    with col2:
        st.subheader("Simulação VAR(1)")
        
        a12 = st.slider("a₁₂ (efeito de Y₂ em Y₁)", -0.5, 0.5, 0.3, 0.1, key="a12")
        a21 = st.slider("a₂₁ (efeito de Y₁ em Y₂)", -0.5, 0.5, 0.2, 0.1, key="a21")
        
        df_var = simulate_var_data(n=150, a12=a12, a21=a21, seed=42)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_var['t'], y=df_var['y1'], name='Y₁'))
        fig.add_trace(go.Scatter(x=df_var['t'], y=df_var['y2'], name='Y₂'))
        fig.update_layout(
            title="Séries Simuladas VAR(1)",
            xaxis_title="Tempo",
            yaxis_title="Valor",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True, key=f"var_sim_{a12}_{a21}")
        
        # Correlação
        corr = np.corrcoef(df_var['y1'], df_var['y2'])[0, 1]
        st.metric("Correlação Y₁, Y₂", f"{corr:.3f}")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa VAR para modelar sistemas onde não sabe a direção causal
    - Foca em previsão e análise de choques (próxima seção)
    """)


def render_section_S7():
    """S7: Ferramentas do VAR: Lags, Granger, IRF e Decomposição da Variância"""
    st.header("🛠️ Ferramentas de Análise VAR")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📏 Seleção de Lags", "🔮 Granger", "📊 IRF", "📈 FEVD"])
    
    with tab1:
        st.subheader("Seleção de Lags: AIC/BIC")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Pergunta:** Quantos lags incluir no VAR?
            
            **Critérios de Informação:**
            - **AIC:** Akaike — penaliza menos
            - **BIC/SBIC:** Bayesian — penaliza mais (prefere parcimônia)
            
            **Regra:** Escolher lag que minimiza o critério
            """)
        
        with col2:
            # Tabela simulada de critérios
            lags_df = pd.DataFrame({
                'Lags': [1, 2, 3, 4],
                'AIC': [-520, -525, -523, -518],
                'BIC': [-510, -512, -505, -495]
            })
            lags_df['Melhor AIC'] = ['', '✓', '', '']
            lags_df['Melhor BIC'] = ['', '✓', '', '']
            
            st.dataframe(lags_df, use_container_width=True, hide_index=True)
            
            st.info("💡 Neste exemplo, lag = 2 é o melhor por ambos os critérios.")
    
    with tab2:
        st.subheader("Causalidade de Granger")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Pergunta:** "Y₂ ajuda a prever Y₁?"
            
            **Teste de Granger:**
            - H₀: Lags de Y₂ não melhoram previsão de Y₁
            - H₁: Lags de Y₂ melhoram previsão de Y₁
            
            **Interpretação:**
            - p < 0.05: Y₂ "Granger-causa" Y₁
            - Não é causalidade no sentido filosófico!
            - É sobre **previsibilidade**
            """)
            
            a12_granger = st.slider("Efeito de Y₂ em Y₁", 0.0, 0.8, 0.4, 0.1, key="granger_a12")
        
        with col2:
            df_var = simulate_var_data(n=200, a12=a12_granger, a21=0.2, seed=42)
            
            granger = granger_test_simple(df_var['y1'].values, df_var['y2'].values)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("F-stat", f"{granger['f_stat']:.2f}")
            col_m2.metric("p-valor", f"{granger['p_value']:.4f}")
            
            if granger['p_value'] < 0.05:
                st.success(f"✅ Rejeita H₀: Y₂ Granger-causa Y₁!")
            else:
                st.info("Não rejeita H₀: Y₂ não melhora previsão de Y₁")
    
    with tab3:
        st.subheader("Impulse Response Function (IRF)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Pergunta:** "Se Y₁ recebe um choque, como Y₁ e Y₂ respondem ao longo do tempo?"
            
            **IRF mostra:**
            - Efeito de um choque unitário
            - Propagação ao longo dos períodos
            - Convergência (volta ao equilíbrio?)
            
            **Uso gerencial:**
            - Choque de juros → Como afeta inflação?
            - Choque de demanda → Como afeta preços?
            """)
            
            choque_em = st.radio("Variável que recebe o choque:", 
                                ["Y₁", "Y₂"], horizontal=True, key="irf_shock")
            
            # Ordenação (importante!)
            ordenacao = st.radio("Ordenação (Cholesky):",
                                ["Y₁ primeiro", "Y₂ primeiro"], 
                                horizontal=True, key="irf_ordem")
        
        with col2:
            # Simular VAR e calcular IRF
            df_var = simulate_var_data(n=200, a12=0.3, a21=0.2, seed=42)
            var_fit = fit_var1_simple(df_var['y1'].values, df_var['y2'].values)
            
            shock_var = 0 if choque_em == "Y₁" else 1
            
            # Ajustar matriz A conforme ordenação
            A = var_fit['A']
            if ordenacao == "Y₂ primeiro":
                A = A[[1, 0], :][:, [1, 0]]
                shock_var = 1 - shock_var
            
            irf = compute_irf(A, periods=20, shock_var=shock_var)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=irf[:, 0], mode='lines+markers', name='Resposta Y₁'))
            fig.add_trace(go.Scatter(y=irf[:, 1], mode='lines+markers', name='Resposta Y₂'))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title=f"IRF: Choque em {choque_em}",
                xaxis_title="Períodos",
                yaxis_title="Resposta",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True, key=f"irf_{choque_em}_{ordenacao}")
            
            st.warning("""
            ⚠️ **Caveat:** A IRF depende da **ordenação** das variáveis!
            Mude a ordenação acima e veja como o gráfico muda.
            """)
    
    with tab4:
        st.subheader("Decomposição da Variância (FEVD)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Pergunta:** "Quanto da variância de Y₁ é explicada por choques em Y₁ vs choques em Y₂?"
            
            **FEVD mostra:**
            - Proporção da variância atribuída a cada fonte
            - Como essa proporção evolui com o horizonte
            
            **Uso gerencial:**
            - Quão "autônoma" é uma variável?
            - Quão dependente de choques externos?
            """)
        
        with col2:
            df_var = simulate_var_data(n=200, a12=0.3, a21=0.2, seed=42)
            var_fit = fit_var1_simple(df_var['y1'].values, df_var['y2'].values)
            
            fevd = compute_fevd(var_fit['A'], periods=20)
            
            # FEVD de Y1
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=fevd[:, 0, 0]*100, mode='lines', 
                                    name='Choques Y₁', fill='tozeroy'))
            fig.add_trace(go.Scatter(y=(fevd[:, 0, 0] + fevd[:, 0, 1])*100, mode='lines',
                                    name='Choques Y₂', fill='tonexty'))
            fig.update_layout(
                title="FEVD de Y₁: Fontes da Variância",
                xaxis_title="Horizonte",
                yaxis_title="% da Variância",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Área inferior = variância explicada por choques próprios")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa Granger para entender previsibilidade entre variáveis
    - Lê IRF para entender propagação de choques
    - Usa FEVD para entender interdependência
    """)


def render_section_S8():
    """S8: Aplicação Prática e Tomada de Decisão"""
    st.header("🏠 Caso: Retornos Imobiliários e Variáveis Macro")
    
    st.markdown("""
    Vamos aplicar VAR para entender como juros, inflação e retornos imobiliários interagem.
    """)
    
    # Dados do caso
    df_case = make_real_estate_case_data(n=100, seed=42)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Contexto do Caso")
        
        st.markdown("""
        **Cenário:** Você é gestor de um fundo imobiliário e quer entender:
        
        1. Como **choques de juros** afetam retornos imobiliários?
        2. Qual o horizonte do impacto?
        3. Quanto da volatilidade dos retornos vem de fatores macro?
        
        **Variáveis:**
        - Juros (taxa básica %)
        - Inflação (% a.a.)
        - Retorno Imobiliário (% a.a.)
        """)
        
        st.dataframe(df_case.head(10).round(2), use_container_width=True)
    
    with col2:
        # Séries
        fig = make_subplots(rows=3, cols=1, 
                           subplot_titles=["Juros", "Inflação", "Retorno Imobiliário"],
                           shared_xaxes=True)
        
        fig.add_trace(go.Scatter(y=df_case['Juros'], name='Juros'), row=1, col=1)
        fig.add_trace(go.Scatter(y=df_case['Inflacao'], name='Inflação'), row=2, col=1)
        fig.add_trace(go.Scatter(y=df_case['Retorno_Imob'], name='Retorno'), row=3, col=1)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Análise: IRF de Choque nos Juros")
    
    # Ajustar VAR simplificado (juros e retorno)
    var_fit = fit_var1_simple(df_case['Juros'].values, df_case['Retorno_Imob'].values)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Pergunta de negócio:**
        > "Se o Banco Central aumenta juros, o que acontece com os retornos imobiliários?"
        
        **IRF mostra:**
        - Efeito instantâneo
        - Persistência ao longo do tempo
        - Quando estabiliza
        """)
        
        # IRF de choque em Juros
        irf = compute_irf(var_fit['A'], periods=15, shock_var=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=irf[:, 0], mode='lines+markers', name='Juros'))
        fig.add_trace(go.Scatter(y=irf[:, 1], mode='lines+markers', name='Retorno Imob.'))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="IRF: Choque de 1% nos Juros",
            xaxis_title="Trimestres",
            yaxis_title="Resposta (%)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Leitura Gerencial da IRF")
        
        impacto_inicial = irf[1, 1]
        impacto_pico = np.min(irf[:, 1])
        periodo_pico = np.argmin(irf[:, 1])
        
        st.metric("Impacto no 1º período", f"{impacto_inicial:.2f}%")
        st.metric("Impacto máximo", f"{impacto_pico:.2f}%", 
                 delta=f"no período {periodo_pico}")
        
        st.markdown("""
        **Interpretação:**
        - Juros ↑ 1% → Retornos imobiliários caem
        - Efeito persiste por vários períodos
        - Convergência gradual ao equilíbrio
        
        **Decisão:**
        - Em ciclo de alta de juros, reduzir exposição imobiliária
        - Esperar X trimestres para estabilização
        """)
    
    st.subheader("⚠️ Limitações e Caveats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Ordenação:**
        - IRF depende da ordem das variáveis
        - Diferentes ordenações = diferentes resultados
        - Justificar com teoria
        """)
    
    with col2:
        st.markdown("""
        **Especificação:**
        - Número de lags importa
        - Variáveis omitidas podem viesar
        - Estacionaridade é necessária
        """)
    
    with col3:
        st.markdown("""
        **Estabilidade:**
        - Parâmetros podem mudar no tempo
        - Crises alteram dinâmicas
        - Usar com cautela em regimes diferentes
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa IRF para timing de alocação setorial
    - Combina análise quantitativa com julgamento sobre regime econômico
    """)


def render_section_S9():
    """S9: Resumo Executivo e Ponte para o Próximo Módulo"""
    st.header("📋 Resumo Executivo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### O que Aprendemos sobre Modelos Multivariados
        
        ✅ **Por que Sistema?**
        - Variáveis frequentemente se influenciam mutuamente
        - Equação única ignora feedback e gera viés
        
        ✅ **Viés de Simultaneidade:**
        - OLS é viesado quando X e Y se determinam juntos
        - Cov(X, ε) ≠ 0 viola exogeneidade
        
        ✅ **Forma Estrutural vs Reduzida:**
        - Estrutural = teoria (mas não estimável diretamente)
        - Reduzida = estimável (depende só de exógenas)
        
        ✅ **Identificação:**
        - Precisa de exclusões (variáveis em uma equação, não em outra)
        - Condição de ordem: exclusões ≥ endógenas - 1
        
        ✅ **IV/2SLS:**
        - Instrumento: afeta X mas não Y diretamente
        - 2SLS: 1º estágio limpa X; 2º estágio estima efeito
        - Trade-off: menos viés, mais variância
        
        ✅ **VAR:**
        - Todas variáveis tratadas como endógenas
        - Flexível, bom para previsão
        - Ferramentas: Granger, IRF, FEVD
        
        ✅ **IRF:**
        - Mostra propagação de choques
        - Essencial para decisões de timing
        - Sensível à ordenação (caveat!)
        """)
    
    with col2:
        st.markdown("### 🧪 Quiz Final")
        
        st.markdown("""
        Um analista quer estimar o efeito de publicidade em vendas.
        Suspeita que o orçamento de publicidade depende das vendas passadas.
        """)
        
        resposta = st.radio(
            "O que você recomendaria?",
            ["OLS é suficiente",
             "Usar IV/2SLS com instrumento válido",
             "VAR é a única opção"],
            key="quiz_final"
        )
        
        if st.button("Ver resposta", key="btn_final"):
            if resposta == "Usar IV/2SLS com instrumento válido":
                st.success("""
                ✅ **Correto!**
                
                Há endogeneidade (feedback vendas → publicidade).
                IV/2SLS corrige o viés se encontrar bom instrumento.
                Exemplos: custo de mídia, regulação de publicidade.
                """)
            else:
                st.error("O cenário tem endogeneidade. OLS será viesado. IV/2SLS é a abordagem correta.")
    
    st.markdown("---")
    
    st.subheader("🔜 Próximo Módulo: Cointegração e Relações de Longo Prazo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Séries não-estacionárias:**
        - Tendências estocásticas
        - Raiz unitária
        - Regressão espúria
        """)
    
    with col2:
        st.markdown("""
        **Cointegração:**
        - Relações de equilíbrio
        - Teste de Johansen
        - Vetores de cointegração
        """)
    
    with col3:
        st.markdown("""
        **VECM:**
        - Correção de erros
        - Curto vs longo prazo
        - Ajuste ao equilíbrio
        """)
    
    st.success("""
    🎓 **Mensagem final:** Quando variáveis interagem, modelos de equação única falham.
    Sistemas (SEM/IV) e VAR permitem capturar essas interações e fazer inferência válida.
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Reconhece quando há endogeneidade e usa IV/2SLS
    - Usa VAR para análise de sistemas macroeconômicos
    - Lê IRF para decisões de timing e alocação
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("🔄 Módulo 7: Modelos Multivariados")
    st.markdown("**Laboratório de Econometria** | SEM, IV/2SLS e VAR")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Explicar por que regressões de equação única falham com **endogeneidade**
        - Introduzir **sistemas de equações simultâneas (SEM)**
        - Ensinar **identificação** e a lógica de exclusões
        - Apresentar **IV/2SLS** como solução para endogeneidade
        - Introduzir **VAR** para dinâmica conjunta em séries temporais
        - Aplicar ferramentas do VAR: **Granger, IRF, FEVD**
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "🔄 Por que Multivariados?",
        "S2": "⚡ Viés de Simultaneidade",
        "S3": "📐 Estrutural vs Reduzida",
        "S4": "🔍 Identificação",
        "S5": "🔧 IV e 2SLS",
        "S6": "📈 VAR",
        "S7": "🛠️ Ferramentas VAR",
        "S8": "🏠 Caso: Imobiliário",
        "S9": "📋 Resumo"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Dica:** Modelos multivariados são essenciais 
    quando variáveis se influenciam mutuamente.
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
            page_title="Módulo 7: Modelos Multivariados",
            page_icon="🔄",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()