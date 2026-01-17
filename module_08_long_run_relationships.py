"""
Laboratório de Econometria - Module 8: Modelling Long-Run Relationships in Finance
Aplicativo educacional interativo para cointegração, ECM/VECM e relações de longo prazo.
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

def make_trending_series(n: int = 200, drift: float = 0.1, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """Gera série não-estacionária (random walk com drift)."""
    np.random.seed(seed)
    e = np.random.normal(0, sigma, n)
    y = np.zeros(n)
    y[0] = 0
    for t in range(1, n):
        y[t] = y[t-1] + drift + e[t]
    return y


def make_stationary_series(n: int = 200, phi: float = 0.7, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """Gera série estacionária AR(1)."""
    np.random.seed(seed)
    e = np.random.normal(0, sigma, n)
    y = np.zeros(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t-1] + e[t]
    return y


def make_spurious_regression_data(n: int = 200, seed: int = 42) -> dict:
    """Gera duas séries não relacionadas para demonstrar regressão espúria."""
    np.random.seed(seed)
    
    # Duas séries completamente independentes, cada uma random walk
    e1 = np.random.normal(0, 1, n)
    e2 = np.random.normal(0, 1, n)
    
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    
    y1[0] = 50
    y2[0] = 30
    
    for t in range(1, n):
        y1[t] = y1[t-1] + 0.1 + e1[t]  # Tendência positiva
        y2[t] = y2[t-1] + 0.08 + e2[t]  # Tendência positiva diferente
    
    return {'y1': y1, 'y2': y2}


def make_cointegrated_series(n: int = 200, beta: float = 1.0, alpha: float = -0.3, 
                             sigma_eq: float = 0.5, seed: int = 42) -> dict:
    """
    Gera duas séries cointegradas.
    y1 e y2 são I(1), mas y1 - beta*y2 é I(0) (estacionário).
    """
    np.random.seed(seed)
    
    # Componente comum (tendência estocástica)
    common = np.zeros(n)
    e_common = np.random.normal(0, 1, n)
    for t in range(1, n):
        common[t] = common[t-1] + e_common[t]
    
    # Erro de equilíbrio (estacionário)
    eq_error = np.zeros(n)
    e_eq = np.random.normal(0, sigma_eq, n)
    for t in range(1, n):
        eq_error[t] = 0.7 * eq_error[t-1] + e_eq[t]
    
    # Séries cointegradas
    y1 = 10 + common + eq_error + np.random.normal(0, 0.3, n)
    y2 = 10 + common / beta + np.random.normal(0, 0.3, n)
    
    # Resíduo de cointegração
    residual = y1 - beta * y2
    
    return {
        'y1': y1,
        'y2': y2,
        'residual': residual,
        'beta': beta,
        'common': common
    }


def adf_test_simple(y: np.ndarray, max_lag: int = 4) -> dict:
    """
    Teste ADF (Augmented Dickey-Fuller) simplificado.
    H0: Série tem raiz unitária (não estacionária)
    H1: Série é estacionária
    """
    n = len(y)
    
    # Diferença
    dy = np.diff(y)
    y_lag = y[:-1]
    
    # Construir regressão: Δy_t = α + γ*y_{t-1} + Σβ_i*Δy_{t-i} + ε
    # Simplificado: só lag 1
    X = np.column_stack([np.ones(len(dy)), y_lag])
    
    # OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ dy
    
    y_hat = X @ beta
    residuals = dy - y_hat
    
    # Estatística t para γ (coeficiente de y_{t-1})
    s2 = np.sum(residuals**2) / (len(dy) - 2)
    se_gamma = np.sqrt(s2 * XtX_inv[1, 1])
    t_stat = beta[1] / se_gamma
    
    # Valores críticos aproximados (MacKinnon)
    # Para n > 100, com constante
    critical_1 = -3.43
    critical_5 = -2.86
    critical_10 = -2.57
    
    # P-valor aproximado usando distribuição normal (simplificação)
    # Na realidade, usa-se a distribuição de Dickey-Fuller
    if t_stat < critical_1:
        p_value = 0.005
    elif t_stat < critical_5:
        p_value = 0.03
    elif t_stat < critical_10:
        p_value = 0.08
    else:
        p_value = 0.15 + 0.1 * (t_stat - critical_10)
        p_value = min(p_value, 0.99)
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'critical_1': critical_1,
        'critical_5': critical_5,
        'critical_10': critical_10,
        'gamma': beta[1]
    }


def kpss_test_simple(y: np.ndarray) -> dict:
    """
    Teste KPSS simplificado.
    H0: Série é estacionária
    H1: Série tem raiz unitária
    """
    n = len(y)
    
    # Remover média
    y_centered = y - np.mean(y)
    
    # Soma parcial dos resíduos
    S = np.cumsum(y_centered)
    
    # Estimador de variância de longo prazo (simplificado)
    # Usando variância amostral
    s2 = np.var(y_centered, ddof=1)
    
    # Estatística KPSS
    kpss_stat = np.sum(S**2) / (n**2 * s2)
    
    # Valores críticos aproximados (com constante)
    critical_1 = 0.739
    critical_5 = 0.463
    critical_10 = 0.347
    
    # P-valor aproximado
    if kpss_stat > critical_1:
        p_value = 0.005
    elif kpss_stat > critical_5:
        p_value = 0.03
    elif kpss_stat > critical_10:
        p_value = 0.08
    else:
        p_value = 0.15
    
    return {
        'kpss_stat': kpss_stat,
        'p_value': p_value,
        'critical_1': critical_1,
        'critical_5': critical_5,
        'critical_10': critical_10
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
    
    # Durbin-Watson
    dw = np.sum(np.diff(residuals)**2) / sse
    
    return {
        'beta': beta,
        'se': se,
        'r_squared': r_squared,
        'residuals': residuals,
        'y_hat': y_hat,
        'dw': dw
    }


def fit_ecm_simple(y1: np.ndarray, y2: np.ndarray, beta_coint: float) -> dict:
    """
    Ajusta modelo ECM (Error Correction Model) simplificado.
    Δy1_t = α + γ*(y1_{t-1} - β*y2_{t-1}) + δ*Δy2_t + ε_t
    """
    n = len(y1)
    
    # Diferenças
    dy1 = np.diff(y1)
    dy2 = np.diff(y2)
    
    # Erro de equilíbrio defasado
    eq_error = y1[:-1] - beta_coint * y2[:-1]
    eq_error_lag = eq_error[:-1]
    
    # Ajustar tamanhos
    dy1 = dy1[1:]
    dy2 = dy2[1:]
    
    # Regressão ECM
    X = np.column_stack([np.ones(len(dy1)), eq_error_lag, dy2])
    ecm = fit_ols_simple(dy1, X)
    
    return {
        'alpha': ecm['beta'][0],
        'gamma': ecm['beta'][1],  # Velocidade de ajuste
        'delta': ecm['beta'][2],  # Efeito de curto prazo
        'se': ecm['se'],
        'r_squared': ecm['r_squared'],
        'residuals': ecm['residuals']
    }


def fit_vecm_simple(y1: np.ndarray, y2: np.ndarray) -> dict:
    """
    Ajusta VECM bivariado simplificado.
    Primeiro estima cointegração, depois ECM para ambas as equações.
    """
    n = len(y1)
    
    # Estimar relação de cointegração via OLS
    X_coint = np.column_stack([np.ones(n), y2])
    coint_reg = fit_ols_simple(y1, X_coint)
    beta_coint = coint_reg['beta'][1]
    
    # Erro de equilíbrio
    eq_error = y1 - coint_reg['beta'][0] - beta_coint * y2
    
    # Testar estacionaridade do erro
    adf_resid = adf_test_simple(eq_error)
    
    # ECM para y1
    ecm_y1 = fit_ecm_simple(y1, y2, beta_coint)
    
    # ECM para y2 (na direção oposta)
    dy2 = np.diff(y2)[1:]
    dy1 = np.diff(y1)[1:]
    eq_error_lag = eq_error[:-2]
    
    X_y2 = np.column_stack([np.ones(len(dy2)), eq_error_lag, dy1])
    ecm_y2 = fit_ols_simple(dy2, X_y2)
    
    return {
        'beta_coint': beta_coint,
        'alpha_coint': coint_reg['beta'][0],
        'adf_residual': adf_resid,
        'gamma_y1': ecm_y1['gamma'],
        'gamma_y2': ecm_y2['beta'][1],
        'eq_error': eq_error,
        'r2_coint': coint_reg['r_squared']
    }


def johansen_test_simple(y1: np.ndarray, y2: np.ndarray) -> dict:
    """
    Teste de Johansen simplificado para cointegração.
    Retorna estatísticas de traço e autovalor máximo.
    """
    n = len(y1)
    
    # Diferenças
    dy1 = np.diff(y1)
    dy2 = np.diff(y2)
    
    # Níveis defasados
    y1_lag = y1[:-1]
    y2_lag = y2[:-1]
    
    # Matriz de dados
    Y = np.column_stack([dy1, dy2])
    Y_lag = np.column_stack([y1_lag, y2_lag])
    
    # Regressão simplificada para obter resíduos
    # (Implementação completa usaria canonical correlations)
    
    # Simular estatísticas baseadas em correlação canônica
    # Correlação entre Y e Y_lag
    corr_matrix = np.corrcoef(Y.T, Y_lag.T)[:2, 2:]
    
    # Autovalores aproximados
    eigenvalues = np.linalg.svd(corr_matrix)[1]**2
    
    # Estatísticas de traço e máximo autovalor
    trace_stat = -n * np.sum(np.log(1 - eigenvalues))
    max_eigen_stat = -n * np.log(1 - eigenvalues[0])
    
    # Valores críticos aproximados (2 variáveis, com constante)
    # r = 0
    trace_crit_r0 = 15.41
    max_crit_r0 = 14.07
    # r = 1
    trace_crit_r1 = 3.76
    max_crit_r1 = 3.76
    
    # Determinar número de vetores de cointegração
    if trace_stat > trace_crit_r0:
        if trace_stat - eigenvalues[0] * n > trace_crit_r1:
            n_coint = 2
        else:
            n_coint = 1
    else:
        n_coint = 0
    
    return {
        'trace_stat_r0': trace_stat,
        'trace_crit_r0': trace_crit_r0,
        'max_eigen_stat_r0': max_eigen_stat,
        'max_crit_r0': max_crit_r0,
        'eigenvalues': eigenvalues,
        'n_cointegration': n_coint
    }


def simulate_ecm_response(gamma: float, n_periods: int = 50, shock: float = 1.0) -> np.ndarray:
    """Simula resposta do ECM a um choque no equilíbrio."""
    response = np.zeros(n_periods)
    response[0] = shock
    
    for t in range(1, n_periods):
        response[t] = response[t-1] * (1 + gamma)  # gamma é negativo
    
    return response


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: Introdução: O Perigo das Relações Espúrias"""
    st.header("⚠️ O Perigo das Relações Espúrias")
    
    st.markdown("""
    **Pergunta de negócio:**
    > "Podemos confiar em correlação alta entre séries que só crescem?"
    
    A resposta é: **frequentemente não!**
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("O Problema")
        
        st.markdown("""
        **Regressão Espúria:**
        - Duas séries com tendência podem parecer correlacionadas
        - Mesmo que não tenham NENHUMA relação real
        - R² alto, t-stats significativos... mas é ilusão!
        
        **Exemplo clássico:**
        - PIB da China vs Preço do queijo na Suíça
        - Ambos crescem → correlação alta
        - Relação causal? Obviamente não!
        """)
        
        seed = st.slider("Seed (mude para ver outros exemplos)", 1, 100, 42, key="seed_spurious")
    
    with col2:
        # Simular regressão espúria
        data = make_spurious_regression_data(n=200, seed=seed)
        
        # Regressão
        X = np.column_stack([np.ones(200), data['y2']])
        reg = fit_ols_simple(data['y1'], X)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("R²", f"{reg['r_squared']:.3f}")
        col_m2.metric("t-stat (β)", f"{reg['beta'][1]/reg['se'][1]:.2f}")
        col_m3.metric("Durbin-Watson", f"{reg['dw']:.2f}")
        
        fig = px.scatter(x=data['y2'], y=data['y1'], opacity=0.5,
                        labels={'x': 'Série Y₂', 'y': 'Série Y₁'})
        
        # Linha de regressão
        x_line = np.linspace(data['y2'].min(), data['y2'].max(), 50)
        y_line = reg['beta'][0] + reg['beta'][1] * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                name=f'OLS: R²={reg["r_squared"]:.2f}', 
                                line=dict(color='red')))
        
        fig.update_layout(title="Regressão Espúria: Séries NÃO Relacionadas!", height=350)
        st.plotly_chart(fig, use_container_width=True, key=f"spurious_{seed}")
    
    st.error(f"""
    🚨 **Alerta:** R² = {reg['r_squared']:.2f} parece ótimo, mas as séries são 
    **completamente independentes**! O Durbin-Watson = {reg['dw']:.2f} (longe de 2) 
    indica autocorrelação nos resíduos — sinal clássico de regressão espúria.
    """)
    
    # Mostrar as séries ao longo do tempo
    with st.expander("📊 Ver séries ao longo do tempo"):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=data['y1'], name='Y₁'))
        fig2.add_trace(go.Scatter(y=data['y2'], name='Y₂'))
        fig2.update_layout(title="Duas Séries Independentes com Tendência", height=300)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        **Note:** Ambas crescem (tendência positiva), mas são geradas por processos 
        completamente separados. A "relação" é apenas coincidência de tendências.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Desconfia de R² alto entre séries com tendência
    - Verifica Durbin-Watson (deve ser ≈ 2)
    - Testa estacionaridade antes de confiar em regressões
    """)


def render_section_S2():
    """S2: Não-Estacionaridade e Raiz Unitária"""
    st.header("📊 Não-Estacionaridade e Raiz Unitária")
    
    st.markdown("""
    Uma série é **não-estacionária** se suas propriedades estatísticas mudam ao longo do tempo.
    O caso mais comum em finanças é o **random walk** (passeio aleatório).
    """)
    
    tab1, tab2 = st.tabs(["🔄 Simulação", "🧪 Testes"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Random Walk: Intuição")
            
            st.markdown("""
            **Random Walk:**
            $$y_t = y_{t-1} + \\varepsilon_t$$
            
            **Interpretação:**
            - Hoje = Ontem + Choque aleatório
            - Choques têm efeito **permanente**
            - A série "lembra" de todos os choques passados
            
            **Série Estacionária (AR(1)):**
            $$y_t = \\phi \\cdot y_{t-1} + \\varepsilon_t, \\ |\\phi| < 1$$
            
            - Choques se dissipam com o tempo
            - Série reverte à média
            """)
            
            tipo = st.radio(
                "Selecione o tipo de série:",
                ["Estacionária (AR(1) com φ=0.7)", "Não-Estacionária (Random Walk)"],
                key="tipo_serie_s2"
            )
        
        with col2:
            n = 200
            np.random.seed(42)
            
            if "Estacionária" in tipo:
                e = np.random.normal(0, 1, n)
                y = np.zeros(n)
                y[0] = e[0]
                for t in range(1, n):
                    y[t] = 0.7 * y[t-1] + e[t]
                titulo = "Série Estacionária AR(1)"
            else:
                np.random.seed(123)
                e = np.random.normal(0, 1, n)
                y = np.zeros(n)
                for t in range(1, n):
                    y[t] = y[t-1] + e[t]
                titulo = "Random Walk (Não-Estacionária)"
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y, mode='lines'))
            fig.add_hline(y=np.mean(y), line_dash="dash", line_color="red",
                         annotation_text=f"Média = {np.mean(y):.2f}")
            fig.update_layout(title=titulo, height=350)
            st.plotly_chart(fig, use_container_width=True, key=f"serie_{tipo}")
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Média (1ª metade)", f"{np.mean(y[:n//2]):.2f}")
            col_m2.metric("Média (2ª metade)", f"{np.mean(y[n//2:]):.2f}")
    
    with tab2:
        st.subheader("Testes de Raiz Unitária")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Teste ADF (Augmented Dickey-Fuller):**
            - H₀: Série tem raiz unitária (não estacionária)
            - H₁: Série é estacionária
            - p < 0.05 → Rejeita H₀ → Série estacionária
            
            **Teste KPSS:**
            - H₀: Série é estacionária
            - H₁: Série tem raiz unitária
            - p < 0.05 → Rejeita H₀ → Série não estacionária
            
            **Estratégia:** Usar ambos para confirmar
            """)
            
            teste_tipo = st.radio(
                "Gerar série para teste:",
                ["Estacionária", "Não-Estacionária"],
                horizontal=True,
                key="teste_tipo"
            )
        
        with col2:
            if teste_tipo == "Estacionária":
                y_test = make_stationary_series(n=200, phi=0.7, seed=42)
            else:
                y_test = make_trending_series(n=200, drift=0.0, seed=42)
            
            adf = adf_test_simple(y_test)
            kpss = kpss_test_simple(y_test)
            
            st.markdown("**Resultados dos Testes:**")
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.metric("ADF t-stat", f"{adf['t_stat']:.3f}")
                st.caption(f"Crítico 5%: {adf['critical_5']:.2f}")
                if adf['t_stat'] < adf['critical_5']:
                    st.success("✅ Rejeita H₀: Estacionária")
                else:
                    st.warning("⚠️ Não rejeita H₀: Raiz unitária")
            
            with col_t2:
                st.metric("KPSS stat", f"{kpss['kpss_stat']:.3f}")
                st.caption(f"Crítico 5%: {kpss['critical_5']:.2f}")
                if kpss['kpss_stat'] > kpss['critical_5']:
                    st.warning("⚠️ Rejeita H₀: Não estacionária")
                else:
                    st.success("✅ Não rejeita H₀: Estacionária")
    
    with st.expander("⚠️ Impacto de Quebras Estruturais"):
        st.markdown("""
        **Cuidado:** Quebras estruturais podem confundir os testes!
        
        - Uma série estacionária com quebra pode parecer ter raiz unitária
        - Testes tradicionais (ADF, KPSS) não consideram quebras
        - Soluções: testes com quebras endógenas (Zivot-Andrews, Lee-Strazicich)
        
        **Na prática:** Se suspeitar de quebra (crise, mudança de regime), 
        divida a amostra ou use testes robustos.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Testa estacionaridade ANTES de modelar
    - Se série tem raiz unitária, diferencia ou usa cointegração
    """)


def render_section_S3():
    """S3: Cointegração: Equilíbrio de Longo Prazo"""
    st.header("🔗 Cointegração: Séries que Andam Juntas")
    
    st.markdown("""
    **Cointegração** ocorre quando duas (ou mais) séries não-estacionárias 
    têm uma combinação linear que É estacionária — um **equilíbrio de longo prazo**.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Intuição")
        
        st.markdown("""
        **Exemplo: Spot e Futuro**
        - Preço spot S_t é I(1) (random walk)
        - Preço futuro F_t é I(1) (random walk)
        - Mas F_t - S_t (base) é I(0) (estacionário)!
        
        **Por quê?**
        - Se divergirem muito, arbitragem corrige
        - Existe um **equilíbrio de longo prazo**
        
        **Outros exemplos:**
        - Taxa de câmbio e preços relativos (PPP)
        - Taxas de juros de diferentes maturidades
        - Preços de ativos relacionados
        """)
        
        beta = st.slider("β (coeficiente de cointegração)", 0.5, 2.0, 1.0, 0.1, key="beta_coint")
        sigma_eq = st.slider("Volatilidade do desvio", 0.2, 1.5, 0.5, 0.1, key="sigma_eq")
    
    with col2:
        # Gerar séries cointegradas
        data = make_cointegrated_series(n=200, beta=beta, sigma_eq=sigma_eq, seed=42)
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=["Séries Y₁ e Y₂", "Resíduo de Cointegração (Y₁ - β·Y₂)"],
                           row_heights=[0.6, 0.4])
        
        fig.add_trace(go.Scatter(y=data['y1'], name='Y₁'), row=1, col=1)
        fig.add_trace(go.Scatter(y=data['y2'], name='Y₂'), row=1, col=1)
        
        fig.add_trace(go.Scatter(y=data['residual'], name='Resíduo', 
                                line=dict(color='green')), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(height=450, showlegend=True)
        st.plotly_chart(fig, use_container_width=True, key=f"coint_{beta}_{sigma_eq}")
        
        # Testar estacionaridade do resíduo
        adf_resid = adf_test_simple(data['residual'])
        st.metric("ADF do Resíduo", f"{adf_resid['t_stat']:.2f}",
                 help="Se < -2.86 (5%), resíduo é estacionário → cointegração!")
        
        if adf_resid['t_stat'] < adf_resid['critical_5']:
            st.success("✅ Resíduo estacionário: Séries são cointegradas!")
        else:
            st.warning("⚠️ Resíduo não-estacionário: Sem evidência de cointegração")
    
    with st.expander("📖 Por que não simplesmente diferenciar?"):
        st.markdown("""
        **Diferenciar elimina a informação de equilíbrio!**
        
        Se você diferencia séries cointegradas e estima VAR em diferenças:
        - Perde a relação de longo prazo
        - Modelo mal especificado
        - Previsões ruins no longo prazo
        
        **Solução:** Usar ECM/VECM que incorpora:
        - Dinâmica de curto prazo (diferenças)
        - Ajuste ao equilíbrio de longo prazo (níveis)
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Identifica pares/grupos de ativos cointegrados
    - Usa desvio do equilíbrio para timing de trades
    - Evita diferenciar quando há cointegração
    """)


def render_section_S4():
    """S4: Modelos de Correção de Erros (ECM/VECM)"""
    st.header("⚡ Modelos de Correção de Erros")
    
    st.markdown("""
    O **ECM (Error Correction Model)** combina dinâmica de curto prazo 
    com ajuste ao equilíbrio de longo prazo.
    """)
    
    tab1, tab2 = st.tabs(["📐 ECM", "📊 VECM"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Modelo ECM")
            
            st.markdown("""
            **Equação do ECM:**
            $$\\Delta y_{1,t} = \\alpha + \\gamma \\cdot (y_{1,t-1} - \\beta \\cdot y_{2,t-1}) + \\delta \\cdot \\Delta y_{2,t} + \\varepsilon_t$$
            
            **Componentes:**
            - **γ (gamma):** Velocidade de ajuste ao equilíbrio
                - Deve ser negativo!
                - |γ| grande → ajuste rápido
            - **β:** Relação de longo prazo
            - **δ:** Efeito de curto prazo
            
            **Teorema de Granger:**
            > Se duas séries são cointegradas, existe representação ECM.
            """)
            
            gamma = st.slider("γ (velocidade de ajuste)", -0.8, -0.05, -0.3, 0.05, key="gamma_ecm")
        
        with col2:
            # Simular resposta a um choque
            response = simulate_ecm_response(gamma, n_periods=30, shock=1.0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=response, mode='lines+markers'))
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         annotation_text="Equilíbrio")
            fig.update_layout(
                title=f"Resposta a Choque no Equilíbrio (γ = {gamma})",
                xaxis_title="Períodos",
                yaxis_title="Desvio do Equilíbrio",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True, key=f"ecm_resp_{gamma}")
            
            # Calcular meia-vida
            if gamma < 0:
                half_life = np.log(0.5) / np.log(1 + gamma)
                st.metric("Meia-vida", f"{half_life:.1f} períodos",
                         help="Tempo para o desvio reduzir pela metade")
    
    with tab2:
        st.subheader("VECM: Modelo Vetorial de Correção de Erros")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **VECM para sistemas:**
            
            $$\\Delta Y_t = \\Pi Y_{t-1} + \\Gamma \\Delta Y_{t-1} + \\varepsilon_t$$
            
            Onde Π = αβ' contém:
            - **α:** Velocidades de ajuste (cada variável)
            - **β:** Vetor de cointegração (equilíbrio)
            
            **Interpretação:**
            - Cada equação tem seu próprio γ
            - Algumas variáveis podem não ajustar (γ ≈ 0)
            - Variável "fracamente exógena" não responde ao desvio
            """)
        
        with col2:
            # Estimar VECM
            data = make_cointegrated_series(n=200, beta=1.0, sigma_eq=0.5, seed=42)
            vecm = fit_vecm_simple(data['y1'], data['y2'])
            
            st.markdown("**Resultados do VECM:**")
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("β (cointegração)", f"{vecm['beta_coint']:.3f}")
            col_m2.metric("R² da relação", f"{vecm['r2_coint']:.3f}")
            
            col_m3, col_m4 = st.columns(2)
            col_m3.metric("γ₁ (ajuste Y₁)", f"{vecm['gamma_y1']:.3f}",
                         help="Negativo = Y₁ corrige desvios")
            col_m4.metric("γ₂ (ajuste Y₂)", f"{vecm['gamma_y2']:.3f}",
                         help="Negativo = Y₂ corrige desvios")
            
            if vecm['gamma_y1'] < 0 and abs(vecm['gamma_y2']) < abs(vecm['gamma_y1']):
                st.info("💡 Y₁ faz a maior parte do ajuste ao equilíbrio.")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa γ para estimar tempo de convergência
    - Identifica qual variável "lidera" (não ajusta) e qual "segue" (ajusta)
    - Baseia estratégias de trading na velocidade de reversão
    """)


def render_section_S5():
    """S5: Johansen e Testes de Hipóteses de Longo Prazo"""
    st.header("🧪 Teste de Johansen")
    
    st.markdown("""
    O **teste de Johansen** determina o número de vetores de cointegração em um sistema 
    e permite testar hipóteses sobre a relação de longo prazo.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Intuição do Teste")
        
        st.markdown("""
        **Pergunta:** Quantas relações de cointegração existem?
        
        **Para 2 variáveis:**
        - r = 0: Nenhuma cointegração (séries independentes)
        - r = 1: Uma relação de equilíbrio
        - r = 2: Ambas estacionárias (raro)
        
        **Testes:**
        - **Traço:** Testa r = 0 vs r ≥ 1, depois r ≤ 1 vs r = 2
        - **Máximo Autovalor:** Testa r = 0 vs r = 1, depois r = 1 vs r = 2
        
        **Interpretação:**
        - Rejeita r = 0 → Há pelo menos 1 vetor de cointegração
        - Não rejeita r ≤ 1 → No máximo 1 vetor
        """)
        
        coint_strength = st.slider("Força da cointegração", 0.1, 1.0, 0.5, 0.1, key="coint_str")
    
    with col2:
        # Gerar dados e testar
        data = make_cointegrated_series(n=200, beta=1.0, sigma_eq=coint_strength, seed=42)
        johansen = johansen_test_simple(data['y1'], data['y2'])
        
        st.markdown("**Resultados do Teste de Johansen:**")
        
        # Tabela de resultados
        results_df = pd.DataFrame({
            'Hipótese': ['r = 0', 'r ≤ 1'],
            'Estatística Traço': [f"{johansen['trace_stat_r0']:.2f}", '-'],
            'Valor Crítico 5%': [f"{johansen['trace_crit_r0']:.2f}", '-'],
            'Decisão': [
                'Rejeita' if johansen['trace_stat_r0'] > johansen['trace_crit_r0'] else 'Não Rejeita',
                '-'
            ]
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        st.metric("Vetores de Cointegração", f"{johansen['n_cointegration']}")
        
        if johansen['n_cointegration'] >= 1:
            st.success("✅ Evidência de cointegração! Pode usar ECM/VECM.")
        else:
            st.warning("⚠️ Sem evidência de cointegração. Considere VAR em diferenças.")
    
    with st.expander("📖 Teste de Restrições no Vetor β"):
        st.markdown("""
        **Após identificar cointegração, podemos testar hipóteses:**
        
        **Exemplo: Spot e Futuro**
        - Teoria: F = S × e^{r×T} → Em logs: f = s + r×T
        - Hipótese: β = 1 (relação 1-para-1)
        - Teste LR: Comparar modelo restrito vs irrestrito
        
        **Exemplo: PPP**
        - Teoria: E = P / P* (câmbio = razão de preços)
        - Em logs: e = p - p*
        - Hipótese: β_p = 1 e β_p* = -1
        
        **Na prática:** Muitas relações teóricas implicam restrições testáveis.
        """)
    
    # Quiz
    st.subheader("🧪 Quiz")
    
    st.markdown("""
    O teste de Johansen para duas séries de preços indica:
    - Estatística de traço para r=0: 18.5 (crítico 5%: 15.41)
    - Estatística de traço para r≤1: 2.3 (crítico 5%: 3.76)
    """)
    
    resposta = st.radio(
        "Qual a conclusão?",
        ["Nenhuma cointegração",
         "Exatamente 1 vetor de cointegração",
         "2 vetores de cointegração"],
        key="quiz_johansen"
    )
    
    if st.button("Ver resposta", key="btn_johansen"):
        if resposta == "Exatamente 1 vetor de cointegração":
            st.success("""
            ✅ **Correto!**
            
            - r=0: 18.5 > 15.41 → Rejeita H₀ → Há pelo menos 1 vetor
            - r≤1: 2.3 < 3.76 → Não rejeita → No máximo 1 vetor
            - Conclusão: Exatamente 1 vetor de cointegração
            """)
        else:
            st.error("Revise: Rejeita r=0 (há cointegração), não rejeita r≤1 (só 1 vetor).")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa Johansen para confirmar cointegração antes de modelar
    - Testa se coeficientes respeitam a teoria econômica
    """)


def render_section_S6():
    """S6: Aplicações e Tomada de Decisão"""
    st.header("💼 Aplicações em Finanças")
    
    tab1, tab2, tab3 = st.tabs(["📈 Previsão", "💹 Trading", "⚠️ Limitações"])
    
    with tab1:
        st.subheader("ECM/VECM vs ARIMA/VAR")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Quando usar ECM/VECM:**
            - Séries cointegradas
            - Interesse em equilíbrio de longo prazo
            - Previsão de médio/longo prazo
            
            **Quando usar VAR em diferenças:**
            - Séries não cointegradas
            - Apenas dinâmica de curto prazo importa
            - Previsão de curto prazo
            
            **Quando usar ARIMA univariado:**
            - Uma única série
            - Sem relação teórica com outras variáveis
            - Previsão operacional simples
            """)
        
        with col2:
            comparacao_df = pd.DataFrame({
                'Característica': ['Usa níveis', 'Usa diferenças', 'Equilíbrio LP', 
                                  'Previsão LP', 'Complexidade'],
                'ECM/VECM': ['✓', '✓', '✓', 'Melhor', 'Alta'],
                'VAR (níveis)': ['✓', '✗', '✗', 'Espúrio?', 'Média'],
                'VAR (dif.)': ['✗', '✓', '✗', 'Ruim', 'Média'],
                'ARIMA': ['✗', '✓', '✗', 'Razoável', 'Baixa']
            })
            st.dataframe(comparacao_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Estratégias de Trading Baseadas em Cointegração")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Pairs Trading / Statistical Arbitrage:**
            
            1. Identificar par cointegrado (ex.: ações do mesmo setor)
            2. Estimar relação de equilíbrio: y₁ = α + β·y₂
            3. Monitorar o spread: z_t = y₁ - β·y₂
            4. Quando z_t > threshold: Short y₁, Long y₂
            5. Quando z_t < -threshold: Long y₁, Short y₂
            6. Fechar quando z_t → 0
            
            **Lógica:** Se cointegração é verdadeira, spread reverte à média.
            """)
        
        with col2:
            # Simular spread e sinais
            data = make_cointegrated_series(n=200, beta=1.0, sigma_eq=0.8, seed=42)
            spread = data['residual']
            
            threshold = 1.0
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=spread, mode='lines', name='Spread'))
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                         annotation_text="Vender")
            fig.add_hline(y=-threshold, line_dash="dash", line_color="green",
                         annotation_text="Comprar")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            
            fig.update_layout(
                title="Spread e Sinais de Trading",
                xaxis_title="Tempo",
                yaxis_title="Spread",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            n_sinais = np.sum(np.abs(spread) > threshold)
            st.metric("Sinais de Trading", f"{n_sinais}")
    
    with tab3:
        st.subheader("Limitações e Cuidados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Limitações Teóricas:**
            
            - Cointegração pode ser instável no tempo
            - Quebras estruturais invalidam relação
            - Amostra curta → testes fracos
            - Número de variáveis → muitos testes
            """)
        
        with col2:
            st.markdown("""
            **Limitações Práticas:**
            
            - **Custos de transação:** Reduzem lucros
            - **Slippage:** Execução diferente do esperado
            - **Funding costs:** Custo de manter posições
            - **Regime shifts:** Relação pode mudar
            """)
        
        st.warning("""
        ⚠️ **Cuidado:** Muitas estratégias de pairs trading falharam durante crises
        porque relações "estáveis" se romperam. Cointegração é estatística, não garantia!
        """)
    
    st.markdown("---")
    
    st.subheader("📋 O que Muda na Decisão?")
    
    decisao_df = pd.DataFrame({
        'Situação': ['Previsão de câmbio LP', 'Hedge de commodities', 
                    'Arbitragem de juros', 'Alocação setorial'],
        'Sem Cointegração': ['VAR em diferenças', 'Correlação histórica',
                            'Análise de curva', 'Correlação de retornos'],
        'Com Cointegração': ['ECM com PPP', 'VECM spot-futuro',
                            'Estrutura a termo cointegrada', 'Pairs dentro do setor']
    })
    st.dataframe(decisao_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Testa cointegração antes de implementar estratégias de reversão
    - Considera custos de transação na avaliação de oportunidades
    - Monitora estabilidade da relação ao longo do tempo
    """)


def render_section_S7():
    """S7: Resumo Executivo e Ponte para o Próximo Módulo"""
    st.header("📋 Resumo Executivo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### O que Aprendemos sobre Relações de Longo Prazo
        
        ✅ **Regressão Espúria:**
        - Séries com tendência podem parecer correlacionadas
        - R² alto não significa relação real
        - Durbin-Watson baixo é sinal de alerta
        
        ✅ **Não-Estacionaridade:**
        - Random walk: choques têm efeito permanente
        - Testes ADF e KPSS para diagnosticar
        - Não diferenciar cegamente — pode perder informação
        
        ✅ **Cointegração:**
        - Séries I(1) com combinação linear I(0)
        - Representa equilíbrio de longo prazo
        - Exemplos: spot-futuro, PPP, estrutura a termo
        
        ✅ **ECM/VECM:**
        - Combina curto prazo (diferenças) e longo prazo (níveis)
        - γ = velocidade de ajuste ao equilíbrio
        - Permite separar quem "lidera" e quem "segue"
        
        ✅ **Teste de Johansen:**
        - Determina número de vetores de cointegração
        - Permite testar hipóteses sobre β
        - Base para especificação do VECM
        
        ✅ **Aplicações:**
        - Previsão de longo prazo superior com ECM
        - Pairs trading baseado em reversão do spread
        - Cuidado com custos e instabilidade
        """)
    
    with col2:
        st.markdown("### 💡 Mensagem-Chave")
        
        st.info("""
        **"Correlação de longo prazo só importa se houver equilíbrio econômico."**
        
        Duas séries podem parecer relacionadas apenas porque ambas crescem.
        
        Cointegração identifica relações com fundamento — onde desvios são temporários.
        """)
        
        st.markdown("### 🧪 Quiz Final")
        
        resposta = st.radio(
            "Se ADF não rejeita raiz unitária para Y₁ e Y₂, mas rejeita para (Y₁ - Y₂):",
            ["Y₁ e Y₂ são estacionárias",
             "Y₁ e Y₂ são cointegradas",
             "Não há relação entre Y₁ e Y₂"],
            key="quiz_final"
        )
        
        if st.button("Ver resposta", key="btn_final"):
            if resposta == "Y₁ e Y₂ são cointegradas":
                st.success("""
                ✅ **Correto!**
                
                - Y₁ é I(1) (não rejeita ADF)
                - Y₂ é I(1) (não rejeita ADF)
                - Y₁ - Y₂ é I(0) (rejeita ADF)
                - Definição de cointegração: combinação linear I(0)!
                """)
            else:
                st.error("A definição de cointegração é exatamente essa: séries I(1) com combinação I(0).")
    
    st.markdown("---")
    
    st.subheader("🔜 Próximo Módulo: Modelagem de Volatilidade")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **ARCH/GARCH:**
        - Volatilidade varia no tempo
        - Clusters de volatilidade
        - Previsão de risco
        """)
    
    with col2:
        st.markdown("""
        **Correlação Dinâmica:**
        - DCC-GARCH
        - Correlações mudam em crises
        - Risco de portfólio
        """)
    
    with col3:
        st.markdown("""
        **Aplicações:**
        - VaR e Expected Shortfall
        - Hedging dinâmico
        - Alocação de risco
        """)
    
    st.success("""
    🎓 **Mensagem final:** Relações de longo prazo em finanças existem, 
    mas precisam de fundamento econômico. Cointegração é a ferramenta 
    para distinguir correlações espúrias de equilíbrios verdadeiros.
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Testa cointegração antes de assumir relações de longo prazo
    - Usa ECM/VECM para previsão e estratégias de reversão
    - Monitora estabilidade das relações ao longo do tempo
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("🔗 Módulo 8: Relações de Longo Prazo em Finanças")
    st.markdown("**Laboratório de Econometria** | Cointegração, ECM e VECM")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Mostrar o perigo das **regressões espúrias**
        - Ensinar a identificar **não-estacionaridade** (ADF, KPSS)
        - Introduzir **cointegração** como equilíbrio de longo prazo
        - Apresentar **ECM/VECM** para modelar curto e longo prazo
        - Explicar o **teste de Johansen** para sistemas
        - Conectar a **decisões** de previsão, trading e hedge
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "⚠️ Relações Espúrias",
        "S2": "📊 Raiz Unitária",
        "S3": "🔗 Cointegração",
        "S4": "⚡ ECM/VECM",
        "S5": "🧪 Teste de Johansen",
        "S6": "💼 Aplicações",
        "S7": "📋 Resumo"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Dica:** Cointegração é fundamental 
    para estratégias de pairs trading e 
    modelagem de equilíbrio em finanças.
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


# =============================================================================
# EXECUÇÃO STANDALONE (para testes)
# =============================================================================

if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Módulo 8: Relações de Longo Prazo",
            page_icon="🔗",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()