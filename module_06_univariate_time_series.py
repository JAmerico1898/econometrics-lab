"""
Laboratório de Econometria - Module 6: Univariate Time Series Modeling and Forecasting
Aplicativo educacional interativo para séries temporais univariadas.
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

@st.cache_data
def simulate_white_noise(n: int = 200, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """Gera ruído branco (série i.i.d. com média 0)."""
    np.random.seed(seed)
    return np.random.normal(0, sigma, n)


@st.cache_data
def simulate_ar(n: int = 200, phi: float = 0.7, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """Simula processo AR(1): y_t = phi * y_{t-1} + e_t."""
    np.random.seed(seed)
    e = np.random.normal(0, sigma, n)
    y = np.zeros(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t-1] + e[t]
    return y


@st.cache_data
def simulate_ma(n: int = 200, theta: float = 0.7, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """Simula processo MA(1): y_t = e_t + theta * e_{t-1}."""
    np.random.seed(seed)
    e = np.random.normal(0, sigma, n + 1)
    y = np.zeros(n)
    for t in range(n):
        y[t] = e[t+1] + theta * e[t]
    return y


@st.cache_data
def simulate_arma(n: int = 200, phi: float = 0.5, theta: float = 0.3, 
                  sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """Simula processo ARMA(1,1): y_t = phi * y_{t-1} + e_t + theta * e_{t-1}."""
    np.random.seed(seed)
    e = np.random.normal(0, sigma, n + 1)
    y = np.zeros(n)
    y[0] = e[1]
    for t in range(1, n):
        y[t] = phi * y[t-1] + e[t+1] + theta * e[t]
    return y


@st.cache_data
def make_nonstationary_rw(n: int = 200, drift: float = 0.0, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """Gera random walk (não estacionário): y_t = y_{t-1} + drift + e_t."""
    np.random.seed(seed)
    e = np.random.normal(0, sigma, n)
    y = np.zeros(n)
    y[0] = 0
    for t in range(1, n):
        y[t] = y[t-1] + drift + e[t]
    return y


def difference_series(y: np.ndarray, d: int = 1) -> np.ndarray:
    """Diferencia a série d vezes."""
    result = y.copy()
    for _ in range(d):
        result = np.diff(result)
    return result


def compute_acf(y: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Calcula a função de autocorrelação (ACF)."""
    n = len(y)
    y_centered = y - np.mean(y)
    var_y = np.var(y)
    
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    
    for k in range(1, max_lag + 1):
        if k < n:
            acf[k] = np.sum(y_centered[k:] * y_centered[:-k]) / (n * var_y)
    
    return acf


def compute_pacf(y: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Calcula a função de autocorrelação parcial (PACF) via Durbin-Levinson."""
    acf = compute_acf(y, max_lag)
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0
    
    if max_lag >= 1:
        pacf[1] = acf[1]
    
    phi = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = acf[1]
    
    for k in range(2, max_lag + 1):
        num = acf[k] - np.sum([phi[k-1, j] * acf[k-j] for j in range(1, k)])
        den = 1 - np.sum([phi[k-1, j] * acf[j] for j in range(1, k)])
        
        if abs(den) < 1e-10:
            pacf[k] = 0
        else:
            phi[k, k] = num / den
            pacf[k] = phi[k, k]
            
            for j in range(1, k):
                phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
    
    return pacf


def fit_ar1_ols(y: np.ndarray) -> dict:
    """Ajusta AR(1) via OLS simples."""
    n = len(y)
    y_lag = y[:-1]
    y_curr = y[1:]
    
    # OLS: y_t = c + phi * y_{t-1}
    X = np.column_stack([np.ones(n-1), y_lag])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y_curr
    
    y_hat = X @ beta
    residuals = y_curr - y_hat
    
    sse = np.sum(residuals**2)
    sigma2 = sse / (n - 3)
    
    # AIC e BIC
    k = 2  # número de parâmetros (c, phi)
    log_lik = -0.5 * (n - 1) * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    aic = -2 * log_lik + 2 * k
    bic = -2 * log_lik + k * np.log(n - 1)
    
    return {
        'const': beta[0],
        'phi': beta[1],
        'residuals': residuals,
        'sigma2': sigma2,
        'aic': aic,
        'bic': bic,
        'y_hat': y_hat
    }


def ljung_box_test(residuals: np.ndarray, max_lag: int = 10) -> dict:
    """Teste de Ljung-Box para autocorrelação nos resíduos."""
    n = len(residuals)
    acf = compute_acf(residuals, max_lag)
    
    # Estatística Q
    q_stat = n * (n + 2) * np.sum([acf[k]**2 / (n - k) for k in range(1, max_lag + 1)])
    
    # P-valor (chi-quadrado com max_lag graus de liberdade)
    p_value = 1 - stats.chi2.cdf(q_stat, max_lag)
    
    return {
        'q_stat': q_stat,
        'p_value': p_value,
        'df': max_lag
    }


def rolling_forecast(y: np.ndarray, window: int = 50, horizon: int = 1) -> dict:
    """Previsão com janela rolante (rolling window)."""
    n = len(y)
    forecasts = []
    actuals = []
    
    for t in range(window, n - horizon + 1):
        # Ajustar modelo na janela
        y_train = y[t-window:t]
        
        # AR(1) simples
        if len(y_train) > 2:
            fit = fit_ar1_ols(y_train)
            # Previsão: y_{t+h} = c + phi * y_t
            forecast = fit['const'] + fit['phi'] * y[t-1]
            forecasts.append(forecast)
            actuals.append(y[t])
    
    return {
        'forecasts': np.array(forecasts),
        'actuals': np.array(actuals)
    }


def recursive_forecast(y: np.ndarray, initial_window: int = 50, horizon: int = 1) -> dict:
    """Previsão com janela expansível (recursive/expanding)."""
    n = len(y)
    forecasts = []
    actuals = []
    
    for t in range(initial_window, n - horizon + 1):
        # Ajustar modelo com todos os dados até t
        y_train = y[:t]
        
        # AR(1) simples
        if len(y_train) > 2:
            fit = fit_ar1_ols(y_train)
            forecast = fit['const'] + fit['phi'] * y[t-1]
            forecasts.append(forecast)
            actuals.append(y[t])
    
    return {
        'forecasts': np.array(forecasts),
        'actuals': np.array(actuals)
    }


def compute_mae_mse(actuals: np.ndarray, forecasts: np.ndarray) -> dict:
    """Calcula MAE e MSE."""
    errors = actuals - forecasts
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }


def exponential_smoothing(y: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Suavização exponencial simples (SES)."""
    n = len(y)
    smoothed = np.zeros(n)
    smoothed[0] = y[0]
    
    for t in range(1, n):
        smoothed[t] = alpha * y[t] + (1 - alpha) * smoothed[t-1]
    
    return smoothed


@st.cache_data
def make_realistic_series(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Gera série temporal 'realista' para demonstração."""
    np.random.seed(seed)
    
    # Componentes
    trend = np.linspace(0, 20, n)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 12)
    ar_component = simulate_ar(n, phi=0.6, sigma=2, seed=seed)
    
    y = 100 + trend + seasonal + ar_component
    
    dates = pd.date_range(start='2010-01-01', periods=n, freq='M')
    
    return pd.DataFrame({
        'Data': dates,
        'Valor': y,
        'Tendencia': 100 + trend,
        'Sazonal': seasonal
    })


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: Introdução: Por que séries temporais univariadas?"""
    st.header("📈 Por que Séries Temporais Univariadas?")
    
    st.markdown("""
    Em muitos problemas de negócio, queremos **prever o futuro** de uma variável 
    usando apenas seu próprio passado — sem precisar de outras variáveis explicativas.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Perguntas de Negócio")
        
        pergunta = st.selectbox(
            "Selecione um caso:",
            ["📦 Prever demanda para gestão de estoque",
             "💰 Prever receita para planejamento orçamentário",
             "📊 Prever retorno/risco para alocação",
             "🏭 Prever produção para capacidade"]
        )
        
        st.markdown("""
        **Por que univariado?**
        - Nem sempre temos variáveis explicativas disponíveis
        - Dados históricos da própria série são abundantes
        - Modelos simples podem ser muito eficazes
        - Rápido de implementar e atualizar
        """)
        
        st.info("""
        💡 **Estrutural vs Univariado:**
        - **Estrutural:** Explica Y usando X₁, X₂... (ex.: vendas = f(preço, marketing))
        - **Univariado:** Prevê Y usando apenas o passado de Y (ex.: vendas_t = f(vendas_{t-1}, vendas_{t-2}...))
        """)
    
    with col2:
        st.subheader("Visualização da Série")
        
        mostrar_tendencia = st.checkbox("Mostrar tendência", value=False)
        mostrar_choques = st.checkbox("Mostrar choques aleatórios", value=False)
        
        df = make_realistic_series(n=120)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df['Data'], y=df['Valor'],
                                mode='lines', name='Série Observada',
                                line=dict(color='blue', width=2)))
        
        if mostrar_tendencia:
            fig.add_trace(go.Scatter(x=df['Data'], y=df['Tendencia'],
                                    mode='lines', name='Tendência',
                                    line=dict(color='red', dash='dash')))
        
        if mostrar_choques:
            choques = df['Valor'] - df['Tendencia'] - df['Sazonal']
            fig.add_trace(go.Scatter(x=df['Data'], y=choques + 100,
                                    mode='lines', name='Choques',
                                    line=dict(color='orange', width=1)))
        
        fig.update_layout(
            title="Série Temporal (ex.: Vendas Mensais)",
            xaxis_title="Data",
            yaxis_title="Valor",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📖 Componentes de uma Série Temporal"):
        st.markdown("""
        Uma série temporal pode ser decomposta em:
        
        1. **Tendência (T):** Movimento de longo prazo (crescimento, declínio)
        2. **Sazonalidade (S):** Padrões que se repetem em intervalos fixos
        3. **Ciclo (C):** Flutuações de médio prazo (ciclos econômicos)
        4. **Irregular/Ruído (I):** Variações aleatórias imprevisíveis
        
        Modelos ARIMA capturam principalmente a estrutura de autocorrelação (como o passado prevê o futuro).
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa modelos univariados para previsões rápidas de curto prazo
    - Combina com julgamento quando há eventos especiais (promoções, crises)
    """)


def render_section_S2():
    """S2: Estacionaridade e Ruído Branco"""
    st.header("📊 Estacionaridade e Ruído Branco")
    
    st.markdown("""
    **Estacionaridade** é a condição-chave para modelos ARMA/ARIMA funcionarem bem.
    Uma série é estacionária se suas propriedades estatísticas não mudam ao longo do tempo.
    """)
    
    tab1, tab2 = st.tabs(["🔄 Estacionaridade", "📡 Ruído Branco"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("O que é Estacionaridade?")
            
            st.markdown("""
            **Série Estacionária:**
            - Média constante ao longo do tempo
            - Variância constante ao longo do tempo
            - Autocorrelação depende apenas da distância entre observações
            
            **Série Não-Estacionária:**
            - Média ou variância mudam com o tempo
            - Ex.: Random walk, série com tendência
            """)
            
            tipo_serie = st.radio(
                "Selecione o tipo:",
                ["Estacionária (AR(1))", "Não-Estacionária (Random Walk)"],
                horizontal=True,
                key="tipo_serie_estac"
            )
            
            drift = 0.0
            if "Random Walk" in tipo_serie:
                drift = st.slider("Drift (tendência)", -0.1, 0.1, 0.0, 0.02, key="drift_rw")
        
        with col2:
            n = 200
            # Gerar série SEM cache para reagir às mudanças
            np.random.seed(42)
            if "Estacionária" in tipo_serie:
                # AR(1) inline
                e = np.random.normal(0, 1, n)
                y = np.zeros(n)
                y[0] = e[0]
                for t in range(1, n):
                    y[t] = 0.7 * y[t-1] + e[t]
                titulo = "Série Estacionária AR(1)"
            else:
                # Random Walk inline
                np.random.seed(123)  # Seed diferente para RW
                e = np.random.normal(0, 1, n)
                y = np.zeros(n)
                y[0] = 0
                for t in range(1, n):
                    y[t] = y[t-1] + drift + e[t]
                titulo = "Random Walk (Não-Estacionária)"
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y, mode='lines', name='Série'))
            fig.add_hline(y=np.mean(y), line_dash="dash", line_color="red",
                         annotation_text=f"Média = {np.mean(y):.2f}")
            fig.update_layout(title=titulo, xaxis_title="Tempo", yaxis_title="Valor", height=350)
            st.plotly_chart(fig, use_container_width=True, key=f"fig_estac_{tipo_serie}_{drift}")
                        
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Média (1ª metade)", f"{np.mean(y[:n//2]):.2f}")
            col_m2.metric("Média (2ª metade)", f"{np.mean(y[n//2:]):.2f}")
            
            if "Random Walk" in tipo_serie:
                st.warning("⚠️ Note como a média muda entre as metades — não-estacionaridade!")
                    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Ruído Branco: O Benchmark")
            
            st.markdown("""
            **Ruído Branco** é uma série completamente imprevisível:
            - Média zero
            - Variância constante
            - Sem autocorrelação (passado não ajuda a prever futuro)
            
            **Por que importa?**
            - É o "nada mais a extrair" dos dados
            - Resíduos de um bom modelo devem parecer ruído branco
            - Se há padrão nos resíduos, o modelo pode melhorar
            """)
            
            sigma_wn = st.slider("Variância do ruído", 0.5, 3.0, 1.0, 0.25)
        
        with col2:
            wn = simulate_white_noise(n=200, sigma=sigma_wn)
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=["Ruído Branco", "ACF do Ruído Branco"])
            
            fig.add_trace(go.Scatter(y=wn, mode='lines', name='Ruído'),
                         row=1, col=1)
            
            # ACF
            acf = compute_acf(wn, max_lag=20)
            fig.add_trace(go.Bar(x=list(range(21)), y=acf, name='ACF'),
                         row=2, col=1)
            # Bandas de confiança
            conf = 1.96 / np.sqrt(len(wn))
            fig.add_hline(y=conf, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=2, col=1)
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ ACF dentro das bandas = sem autocorrelação significativa")
    
    with st.expander("⚠️ O que acontece se eu ignorar não-estacionaridade?"):
        st.markdown("""
        **Problemas graves:**
        
        1. **Regressão espúria:** Correlações altas entre séries não relacionadas
        2. **Testes inválidos:** t-stats e F-stats não têm distribuição padrão
        3. **Previsões ruins:** Modelo não captura a dinâmica correta
        4. **R² inflado:** Parece bom ajuste, mas é ilusão
        
        **Solução:** Diferenciar a série (ARIMA com d > 0) para torná-la estacionária.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Verifica se a série tem tendência ou variância crescente antes de modelar
    - Se não for estacionária, diferencia (ou usa ARIMA com d=1)
    """)


def render_section_S3():
    """S3: Componentes AR e MA (memória vs choques)"""
    st.header("🔄 Processos AR e MA")
    
    st.markdown("""
    Os dois blocos fundamentais de séries temporais:
    - **AR (Autoregressivo):** O valor atual depende dos valores passados (memória)
    - **MA (Média Móvel):** O valor atual depende dos choques passados (persistência de choques)
    """)
    
    tab1, tab2, tab3 = st.tabs(["📈 AR(1)", "📉 MA(1)", "⚖️ Comparação"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Processo AR(1)")
            
            st.latex(r"y_t = \phi \cdot y_{t-1} + \varepsilon_t")
            
            st.markdown("""
            **Interpretação:**
            - φ (phi) controla a "memória" da série
            - |φ| < 1: Série estacionária (volta à média)
            - φ > 0: Inércia positiva (valores altos seguem altos)
            - φ < 0: Oscilação (valores alternam)
            """)
            
            phi = st.slider("φ (phi)", -0.95, 0.95, 0.7, 0.05, key="phi_ar")
            
            if abs(phi) >= 1:
                st.error("⚠️ |φ| ≥ 1 torna a série não-estacionária!")
        
        with col2:
            y_ar = simulate_ar(n=200, phi=phi, sigma=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_ar, mode='lines', name='AR(1)'))
            fig.update_layout(
                title=f"AR(1) com φ = {phi}",
                xaxis_title="Tempo",
                yaxis_title="Valor",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ACF teórica: decai exponencialmente
            acf_ar = compute_acf(y_ar, max_lag=15)
            
            fig2 = go.Figure(go.Bar(x=list(range(16)), y=acf_ar))
            fig2.add_hline(y=1.96/np.sqrt(200), line_dash="dash", line_color="red")
            fig2.add_hline(y=-1.96/np.sqrt(200), line_dash="dash", line_color="red")
            fig2.update_layout(title="ACF do AR(1)", height=250)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Processo MA(1)")
            
            st.latex(r"y_t = \varepsilon_t + \theta \cdot \varepsilon_{t-1}")
            
            st.markdown("""
            **Interpretação:**
            - θ (theta) controla a persistência dos choques
            - Choques afetam o período atual E o próximo
            - ACF corta abruptamente após lag 1
            - Memória "curta" — efeito desaparece após q períodos
            """)
            
            theta = st.slider("θ (theta)", -0.95, 0.95, 0.7, 0.05, key="theta_ma")
        
        with col2:
            y_ma = simulate_ma(n=200, theta=theta, sigma=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_ma, mode='lines', name='MA(1)'))
            fig.update_layout(
                title=f"MA(1) com θ = {theta}",
                xaxis_title="Tempo",
                yaxis_title="Valor",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            acf_ma = compute_acf(y_ma, max_lag=15)
            
            fig2 = go.Figure(go.Bar(x=list(range(16)), y=acf_ma))
            fig2.add_hline(y=1.96/np.sqrt(200), line_dash="dash", line_color="red")
            fig2.add_hline(y=-1.96/np.sqrt(200), line_dash="dash", line_color="red")
            fig2.update_layout(title="ACF do MA(1)", height=250)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("AR vs MA: Lado a Lado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            phi_comp = st.slider("φ para AR(1)", 0.1, 0.9, 0.7, 0.1, key="phi_comp")
        with col2:
            theta_comp = st.slider("θ para MA(1)", 0.1, 0.9, 0.7, 0.1, key="theta_comp")
        
        y_ar_comp = simulate_ar(n=200, phi=phi_comp, sigma=1, seed=123)
        y_ma_comp = simulate_ma(n=200, theta=theta_comp, sigma=1, seed=123)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"AR(1) φ={phi_comp}", f"MA(1) θ={theta_comp}"])
        
        fig.add_trace(go.Scatter(y=y_ar_comp, mode='lines', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(y=y_ma_comp, mode='lines', line=dict(color='green')), row=1, col=2)
        
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        | Característica | AR | MA |
        |----------------|----|----|
        | Memória | Longa (decai gradualmente) | Curta (corta em q) |
        | ACF | Decai exponencialmente | Corta após lag q |
        | PACF | Corta após lag p | Decai exponencialmente |
        | Interpretação | Inércia, tendência local | Choques temporários |
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - AR: Quando o passado recente influencia persistentemente (ex.: vendas com inércia)
    - MA: Quando choques têm efeito temporário (ex.: promoção pontual)
    """)


def render_section_S4():
    """S4: ARMA/ARIMA e a ideia de integração (I)"""
    st.header("🔗 ARMA e ARIMA")
    
    st.markdown("""
    **ARMA(p,q)** combina AR e MA. **ARIMA(p,d,q)** adiciona diferenciação para séries não-estacionárias.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("ARMA(1,1)")
        
        st.latex(r"y_t = \phi \cdot y_{t-1} + \varepsilon_t + \theta \cdot \varepsilon_{t-1}")
        
        phi_arma = st.slider("φ (AR)", 0.0, 0.9, 0.5, 0.1, key="phi_arma")
        theta_arma = st.slider("θ (MA)", 0.0, 0.9, 0.3, 0.1, key="theta_arma")
        
        y_arma = simulate_arma(n=200, phi=phi_arma, theta=theta_arma, sigma=1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_arma, mode='lines'))
        fig.update_layout(title=f"ARMA(1,1): φ={phi_arma}, θ={theta_arma}", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ARIMA: Diferenciação")
        
        st.markdown("""
        **Quando a série não é estacionária:**
        - Diferenciar: Δyₜ = yₜ - yₜ₋₁
        - ARIMA(p,d,q): d = ordem de diferenciação
        - ARIMA(1,1,1) = ARMA(1,1) na série diferenciada
        """)
        
        # Gerar série não-estacionária
        y_rw = make_nonstationary_rw(n=200, drift=0.1, sigma=1)
        y_diff = difference_series(y_rw, d=1)
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=["Série Original (Random Walk)", "Série Diferenciada"])
        
        fig.add_trace(go.Scatter(y=y_rw, mode='lines', name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(y=y_diff, mode='lines', name='Δy'), row=2, col=1)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ Após diferenciar, a série parece estacionária!")
    
    with st.expander("📖 Notação ARIMA(p,d,q)"):
        st.markdown("""
        - **p:** Ordem do componente AR (quantos lags de y)
        - **d:** Ordem de diferenciação (quantas vezes diferenciar)
        - **q:** Ordem do componente MA (quantos lags do erro)
        
        **Exemplos comuns:**
        - ARIMA(1,0,0) = AR(1)
        - ARIMA(0,0,1) = MA(1)
        - ARIMA(1,1,1) = ARMA(1,1) na série diferenciada
        - ARIMA(0,1,0) = Random walk
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Se a série tem tendência clara, usa d=1 (diferenciação)
    - Combina AR e MA conforme os padrões de ACF/PACF
    """)


def render_section_S5():
    """S5: Box-Jenkins na prática (ACF/PACF, diagnóstico, parcimônia)"""
    st.header("🔧 Metodologia Box-Jenkins")
    
    st.markdown("""
    O processo sistemático para construir modelos ARIMA:
    **Identificar → Estimar → Diagnosticar → (Repetir se necessário)**
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 ACF/PACF", "🔍 Diagnóstico", "📏 Seleção de Modelo"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Identificação via ACF/PACF")
            
            processo = st.selectbox(
                "Gerar série de tipo:",
                ["AR(1) com φ=0.7", "MA(1) com θ=0.7", "ARMA(1,1)", "Ruído Branco"]
            )
            
            if "AR(1)" in processo:
                y = simulate_ar(n=200, phi=0.7, sigma=1)
            elif "MA(1)" in processo:
                y = simulate_ma(n=200, theta=0.7, sigma=1)
            elif "ARMA" in processo:
                y = simulate_arma(n=200, phi=0.5, theta=0.3, sigma=1)
            else:
                y = simulate_white_noise(n=200, sigma=1)
            
            st.markdown("""
            **Regras de identificação:**
            
            | Processo | ACF | PACF |
            |----------|-----|------|
            | AR(p) | Decai | Corta após lag p |
            | MA(q) | Corta após lag q | Decai |
            | ARMA(p,q) | Decai | Decai |
            """)
        
        with col2:
            acf = compute_acf(y, max_lag=15)
            pacf = compute_pacf(y, max_lag=15)
            conf = 1.96 / np.sqrt(len(y))
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=["ACF", "PACF"])
            
            fig.add_trace(go.Bar(x=list(range(16)), y=acf, name='ACF'), row=1, col=1)
            fig.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=1)
            
            fig.add_trace(go.Bar(x=list(range(16)), y=pacf, name='PACF'), row=2, col=1)
            fig.add_hline(y=conf, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=2, col=1)
            
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Diagnóstico: Resíduos são Ruído Branco?")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Um bom modelo deve ter resíduos que:**
            - Não mostram autocorrelação
            - Parecem ruído branco
            
            **Teste de Ljung-Box:**
            - H₀: Resíduos são ruído branco
            - H₁: Há autocorrelação nos resíduos
            - p-valor < 0.05 → modelo inadequado
            """)
            
            # Ajustar AR(1) e verificar resíduos
            y_test = simulate_ar(n=200, phi=0.7, sigma=1, seed=456)
            fit = fit_ar1_ols(y_test)
            
            lb_test = ljung_box_test(fit['residuals'], max_lag=10)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Q-stat (Ljung-Box)", f"{lb_test['q_stat']:.2f}")
            col_m2.metric("p-valor", f"{lb_test['p_value']:.4f}")
            
            if lb_test['p_value'] > 0.05:
                st.success("✅ Não rejeita H₀: Resíduos parecem ruído branco")
            else:
                st.error("❌ Rejeita H₀: Há padrão nos resíduos — modelo pode melhorar")
        
        with col2:
            # ACF dos resíduos
            acf_res = compute_acf(fit['residuals'], max_lag=15)
            conf = 1.96 / np.sqrt(len(fit['residuals']))
            
            fig = go.Figure(go.Bar(x=list(range(16)), y=acf_res))
            fig.add_hline(y=conf, line_dash="dash", line_color="red")
            fig.add_hline(y=-conf, line_dash="dash", line_color="red")
            fig.update_layout(title="ACF dos Resíduos", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Seleção: AIC e BIC")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Critérios de Informação:**
            - Penalizam complexidade para evitar overfitting
            - **AIC:** Akaike Information Criterion
            - **BIC/SBIC:** Bayesian IC (penaliza mais)
            
            **Regra:** Menor valor = melhor modelo
            
            **Parcimônia:**
            - Prefira modelos mais simples
            - Não adicione parâmetros sem ganho significativo
            """)
        
        with col2:
            # Comparar modelos fictícios
            st.markdown("**Comparação de Modelos Candidatos:**")
            
            y_comp = simulate_arma(n=200, phi=0.6, theta=0.3, sigma=1, seed=789)
            
            # AR(1)
            fit_ar = fit_ar1_ols(y_comp)
            
            modelos_df = pd.DataFrame({
                'Modelo': ['AR(1)', 'AR(2)*', 'ARMA(1,1)*'],
                'AIC': [fit_ar['aic'], fit_ar['aic'] - 2, fit_ar['aic'] - 5],
                'BIC': [fit_ar['bic'], fit_ar['bic'] + 1, fit_ar['bic'] - 2],
                'Ljung-Box p': [0.45, 0.52, 0.78]
            })
            modelos_df['Ranking AIC'] = modelos_df['AIC'].rank().astype(int)
            
            st.dataframe(modelos_df.round(2), use_container_width=True, hide_index=True)
            st.caption("* Valores simulados para ilustração")
            
            st.info("💡 ARMA(1,1) tem menor AIC, mas AR(1) é mais simples. Avalie o trade-off!")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa ACF/PACF para escolher ordens p e q
    - Verifica Ljung-Box para garantir que não há padrão nos resíduos
    - Escolhe modelo pelo AIC/BIC, preferindo parcimônia
    """)


def render_section_S6():
    """S6: Previsão e Avaliação (o placar do modelo)"""
    st.header("🎯 Previsão e Avaliação")
    
    st.markdown("""
    O objetivo final é prever bem. Como avaliar se o modelo funciona na prática?
    """)
    
    tab1, tab2, tab3 = st.tabs(["📈 Previsão", "🔄 Backtesting", "📊 Métricas"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Tipos de Previsão")
            
            st.markdown("""
            **One-step-ahead:**
            - Prever apenas o próximo período
            - Usa todos os dados até t para prever t+1
            - Mais preciso, mas limitado
            
            **Multi-step:**
            - Prever vários períodos à frente (t+1, t+2, ... t+h)
            - Incerteza cresce com o horizonte
            - Necessário para planejamento de médio prazo
            """)
            
            horizonte = st.slider("Horizonte de previsão", 1, 20, 5)
        
        with col2:
            # Simular previsão
            y = simulate_ar(n=150, phi=0.7, sigma=1, seed=42)
            fit = fit_ar1_ols(y)
            
            # Previsões multi-step
            previsoes = np.zeros(horizonte)
            ultimo = y[-1]
            for h in range(horizonte):
                previsoes[h] = fit['const'] + fit['phi'] * ultimo
                ultimo = previsoes[h]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y[-50:], mode='lines', name='Histórico'))
            fig.add_trace(go.Scatter(x=list(range(50, 50+horizonte)), y=previsoes,
                                    mode='lines+markers', name='Previsão',
                                    line=dict(color='red', dash='dash')))
            fig.update_layout(title=f"Previsão {horizonte} passos à frente", height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Backtesting: Rolling vs Recursive")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Rolling Window:**
            - Janela de tamanho fixo que "rola" no tempo
            - Ex.: Sempre usar últimos 50 períodos
            - Mais adaptativo a mudanças
            
            **Recursive (Expanding):**
            - Janela que cresce com o tempo
            - Usa todos os dados disponíveis até t
            - Mais dados = estimativas mais estáveis
            """)
            
            metodo = st.radio("Método de backtesting:", 
                             ["Rolling Window", "Recursive (Expanding)"],
                             horizontal=True)
            
            window_size = st.slider("Tamanho da janela inicial", 30, 80, 50)
        
        with col2:
            y = simulate_ar(n=150, phi=0.7, sigma=1, seed=42)
            
            if metodo == "Rolling Window":
                result = rolling_forecast(y, window=window_size)
            else:
                result = recursive_forecast(y, initial_window=window_size)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=result['actuals'], mode='lines', name='Real'))
            fig.add_trace(go.Scatter(y=result['forecasts'], mode='lines', 
                                    name='Previsão', line=dict(dash='dash')))
            fig.update_layout(title=f"Backtest ({metodo})", height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Métricas de Avaliação")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **MAE (Mean Absolute Error):**
            - Média dos erros absolutos
            - Na mesma unidade de Y
            - Interpretação direta
            
            **MSE (Mean Squared Error):**
            - Média dos erros ao quadrado
            - Penaliza mais erros grandes
            - Sensível a outliers
            
            **RMSE:** Raiz do MSE (mesma unidade de Y)
            """)
            
            st.latex(r"MAE = \frac{1}{n}\sum|y_t - \hat{y}_t|")
            st.latex(r"MSE = \frac{1}{n}\sum(y_t - \hat{y}_t)^2")
        
        with col2:
            # Calcular métricas do backtest
            if len(result['forecasts']) > 0:
                metrics = compute_mae_mse(result['actuals'], result['forecasts'])
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("MAE", f"{metrics['mae']:.3f}")
                col_m2.metric("MSE", f"{metrics['mse']:.3f}")
                col_m3.metric("RMSE", f"{metrics['rmse']:.3f}")
                
                # Gráfico de erros
                errors = result['actuals'] - result['forecasts']
                fig = px.histogram(errors, nbins=30, title="Distribuição dos Erros de Previsão")
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Faz backtest antes de usar modelo em produção
    - Escolhe métrica conforme o custo do erro (MAE para geral, MSE se erros grandes são críticos)
    """)


def render_section_S7():
    """S7: Alternativas e Extensões (visão de gestor)"""
    st.header("🔄 Alternativas e Extensões")
    
    tab1, tab2 = st.tabs(["📉 Suavização Exponencial", "🔗 VAR e Granger"])
    
    with tab1:
        st.subheader("Suavização Exponencial Simples")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Alternativa simples a ARIMA:**
            
            $$\\hat{y}_{t+1} = \\alpha \\cdot y_t + (1-\\alpha) \\cdot \\hat{y}_t$$
            
            - α (alpha) controla o peso do último valor vs histórico
            - α próximo de 1: Segue de perto os dados recentes
            - α próximo de 0: Média mais suave, reage devagar
            
            **Vantagens:**
            - Muito simples de implementar
            - Funciona bem para séries sem tendência forte
            - Fácil de explicar para não-técnicos
            """)
            
            alpha = st.slider("α (alpha)", 0.05, 0.95, 0.3, 0.05)
        
        with col2:
            y = simulate_ar(n=100, phi=0.5, sigma=2, seed=42)
            smoothed = exponential_smoothing(y, alpha=alpha)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y, mode='lines', name='Original', opacity=0.7))
            fig.add_trace(go.Scatter(y=smoothed, mode='lines', name=f'Suavizado (α={alpha})',
                                    line=dict(color='red', width=2)))
            fig.update_layout(title="Suavização Exponencial", height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparar erro
            mae_naive = np.mean(np.abs(np.diff(y)))  # Previsão ingênua: y_{t+1} = y_t
            mae_ses = np.mean(np.abs(y[1:] - smoothed[:-1]))
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("MAE (Ingênuo)", f"{mae_naive:.3f}")
            col_m2.metric("MAE (SES)", f"{mae_ses:.3f}")
    
    with tab2:
        st.subheader("VAR e Causalidade de Granger")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **VAR (Vector Autoregression):**
            - Modelo multivariado: várias séries se influenciam mutuamente
            - Ex.: Juros, câmbio e inflação juntos
            - Cada variável é AR + lags das outras
            
            **Causalidade de Granger:**
            - "X Granger-causa Y" se lags de X ajudam a prever Y
            - Não é causalidade no sentido filosófico!
            - É sobre **previsibilidade**, não mecanismo
            
            **Quando usar:**
            - Quando variáveis claramente interagem
            - Para entender dinâmicas de sistema
            - Para previsões condicionais
            """)
            
            st.info("""
            💡 **Exemplo:** Taxa de juros "Granger-causa" preços de imóveis?
            Se lags de juros melhoram a previsão de preços, sim!
            """)
        
        with col2:
            # Mini-simulação de Granger
            st.markdown("**Mini-Simulação: X ajuda a prever Y?**")
            
            np.random.seed(42)
            n = 100
            x = simulate_ar(n=n, phi=0.7, sigma=1)
            
            # Y depende de seu lag + lag de X
            granger_effect = st.slider("Efeito de X_{t-1} sobre Y_t", 0.0, 0.8, 0.4, 0.1)
            
            y = np.zeros(n)
            y[0] = np.random.normal(0, 1)
            for t in range(1, n):
                y[t] = 0.5 * y[t-1] + granger_effect * x[t-1] + np.random.normal(0, 1)
            
            # Comparar previsão com e sem X
            # Modelo 1: só lag de Y
            mae_sem_x = np.mean(np.abs(y[2:] - 0.5 * y[1:-1]))
            # Modelo 2: lag de Y + lag de X
            mae_com_x = np.mean(np.abs(y[2:] - (0.5 * y[1:-1] + granger_effect * x[1:-1])))
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("MAE sem X", f"{mae_sem_x:.3f}")
            col_m2.metric("MAE com X", f"{mae_com_x:.3f}",
                         delta=f"{(mae_com_x/mae_sem_x - 1)*100:.1f}%")
            
            if granger_effect > 0.2:
                st.success(f"✅ X ajuda a prever Y! Incluir X reduz o erro.")
            else:
                st.info("X tem pouco efeito sobre a previsão de Y.")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa suavização exponencial para previsões rápidas e simples
    - Considera VAR quando múltiplas variáveis de interesse interagem
    """)


def render_section_S8():
    """S8: Resumo Executivo e Ponte para o Próximo Módulo"""
    st.header("📋 Resumo Executivo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### O que Aprendemos sobre Séries Temporais Univariadas
        
        ✅ **Estacionaridade:**
        - Condição essencial para ARMA/ARIMA
        - Média e variância constantes no tempo
        - Se não for estacionária, diferencie (ARIMA com d>0)
        
        ✅ **Componentes AR e MA:**
        - AR(p): Memória — valor atual depende dos passados
        - MA(q): Choques — efeito de inovações passadas
        - ARMA/ARIMA combina ambos
        
        ✅ **Box-Jenkins:**
        - ACF/PACF para identificar ordens p e q
        - Ljung-Box para verificar se resíduos são ruído branco
        - AIC/BIC para selecionar modelo (parcimônia)
        
        ✅ **Previsão e Avaliação:**
        - One-step vs multi-step
        - Rolling vs recursive backtesting
        - MAE/MSE como métricas de desempenho
        
        ✅ **Alternativas:**
        - Suavização exponencial: simples e eficaz
        - VAR: quando variáveis interagem
        - Granger: X ajuda a prever Y?
        """)
    
    with col2:
        st.markdown("### 🧪 Quiz Final")
        
        st.markdown("""
        Uma série de vendas mensais mostra ACF que decai lentamente e PACF que corta após lag 2.
        """)
        
        resposta = st.radio(
            "Qual modelo você sugeriria?",
            ["MA(2)", "AR(2)", "ARMA(1,1)", "Random Walk"],
            key="quiz_final"
        )
        
        if st.button("Ver resposta", key="btn_final"):
            if resposta == "AR(2)":
                st.success("""
                ✅ **Correto!** 
                - ACF decai = componente AR
                - PACF corta após lag 2 = AR(2)
                
                A assinatura de AR(p) é exatamente: ACF decai, PACF corta em p.
                """)
            else:
                st.error("""
                AR(2) é a resposta. Lembre-se:
                - AR: ACF decai, PACF corta
                - MA: ACF corta, PACF decai
                """)
    
    st.markdown("---")
    
    st.subheader("🔜 Próximo Módulo: Modelos Multivariados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **VAR:**
        - Múltiplas séries juntas
        - Impulso-resposta
        - Decomposição de variância
        """)
    
    with col2:
        st.markdown("""
        **Cointegração:**
        - Relações de longo prazo
        - Séries não-estacionárias
        - Modelo de correção de erros
        """)
    
    with col3:
        st.markdown("""
        **Aplicações:**
        - Macroeconomia
        - Finanças
        - Política monetária
        """)
    
    st.success("""
    🎓 **Mensagem final:** Modelos univariados são surpreendentemente poderosos para previsão de curto prazo.
    Comece simples, valide com backtesting, e só adicione complexidade se necessário.
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa ARIMA para previsões operacionais (demanda, estoque, caixa)
    - Combina modelo estatístico com julgamento sobre eventos especiais
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("📈 Módulo 6: Séries Temporais Univariadas")
    st.markdown("**Laboratório de Econometria** | ARIMA, Previsão e Box-Jenkins")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Distinguir modelos **estruturais** de **univariados** de séries temporais
        - Ensinar **estacionaridade** e por que é condição-chave
        - Explicar processos **AR, MA, ARMA e ARIMA**
        - Aplicar **Box-Jenkins**: ACF/PACF, Ljung-Box, AIC/BIC
        - Construir e avaliar **previsões** com backtesting
        - Apresentar alternativas: suavização exponencial, VAR/Granger
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "📈 Por que Séries Univariadas?",
        "S2": "📊 Estacionaridade",
        "S3": "🔄 AR e MA",
        "S4": "🔗 ARMA/ARIMA",
        "S5": "🔧 Box-Jenkins",
        "S6": "🎯 Previsão e Avaliação",
        "S7": "🔄 Alternativas e Extensões",
        "S8": "📋 Resumo e Próximos Passos"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Dica:** Séries temporais são fundamentais 
    para previsão em finanças e operações.
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
            page_title="Módulo 6: Séries Temporais Univariadas",
            page_icon="📈",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()