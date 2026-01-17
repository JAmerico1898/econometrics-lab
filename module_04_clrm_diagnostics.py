"""
Laboratório de Econometria - Module 4: Assumptions and Diagnostic Tests of CLRM
Aplicativo educacional interativo para diagnósticos do modelo de regressão linear.
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
# FUNÇÕES AUXILIARES PARA GERAÇÃO DE DADOS
# =============================================================================

@st.cache_data
def make_hetero_data(n: int = 200, hetero_intensity: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Gera dados com heterocedasticidade controlada."""
    np.random.seed(seed)
    x = np.random.uniform(10, 100, n)
    
    # Erro com variância que cresce com x
    if hetero_intensity > 0:
        sigma = 2 + hetero_intensity * x / 10
    else:
        sigma = np.full(n, 5.0)
    
    erro = np.random.normal(0, 1, n) * sigma
    y = 10 + 0.5 * x + erro
    
    return pd.DataFrame({'x': x, 'y': y, 'sigma': sigma})


@st.cache_data
def make_autocorr_ts_data(n: int = 100, rho: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Gera série temporal com erro AR(1) controlado."""
    np.random.seed(seed)
    
    # Tendência temporal
    t = np.arange(1, n + 1)
    x = t + np.random.normal(0, 5, n)
    
    # Erro AR(1): u_t = rho * u_{t-1} + e_t
    e = np.random.normal(0, 2, n)
    u = np.zeros(n)
    u[0] = e[0]
    for i in range(1, n):
        u[i] = rho * u[i-1] + e[i]
    
    y = 5 + 0.3 * x + u
    
    return pd.DataFrame({'t': t, 'x': x, 'y': y, 'u': u})


@st.cache_data
def make_collinear_data(n: int = 200, corr: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Gera dados com multicolinearidade controlada."""
    np.random.seed(seed)
    
    # x1 é independente
    x1 = np.random.normal(50, 10, n)
    
    # x2 é correlacionado com x1
    noise = np.random.normal(0, 10 * np.sqrt(1 - corr**2), n) if abs(corr) < 1 else np.zeros(n)
    x2 = corr * (x1 - 50) + 50 + noise
    
    # y depende de ambos
    erro = np.random.normal(0, 5, n)
    y = 10 + 2 * x1 + 3 * x2 + erro
    
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


@st.cache_data
def make_nonnormal_data(n: int = 200, outlier_pct: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Gera dados com possíveis outliers/eventos extremos."""
    np.random.seed(seed)
    
    x = np.random.uniform(10, 90, n)
    erro = np.random.normal(0, 5, n)
    
    # Adicionar outliers
    n_outliers = int(n * outlier_pct / 100)
    if n_outliers > 0:
        outlier_idx = np.random.choice(n, n_outliers, replace=False)
        erro[outlier_idx] = np.random.choice([-1, 1], n_outliers) * np.random.uniform(20, 40, n_outliers)
    
    y = 15 + 0.8 * x + erro
    
    return pd.DataFrame({'x': x, 'y': y})


@st.cache_data
def make_structural_break_data(n: int = 100, break_point: int = 50, 
                                has_break: bool = False, seed: int = 42) -> pd.DataFrame:
    """Gera dados com possível quebra estrutural."""
    np.random.seed(seed)
    
    t = np.arange(1, n + 1)
    x = np.random.uniform(10, 50, n)
    erro = np.random.normal(0, 3, n)
    
    if has_break:
        # Antes da quebra
        y1 = 10 + 1.0 * x[:break_point] + erro[:break_point]
        # Depois da quebra (coeficientes mudam)
        y2 = 25 + 0.3 * x[break_point:] + erro[break_point:]
        y = np.concatenate([y1, y2])
    else:
        y = 10 + 1.0 * x + erro
    
    regime = np.array(['Antes'] * break_point + ['Depois'] * (n - break_point))
    
    return pd.DataFrame({'t': t, 'x': x, 'y': y, 'regime': regime})


@st.cache_data
def make_ratings_case_data(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Gera dados sintéticos de ratings soberanos."""
    np.random.seed(seed)
    
    # Variáveis macroeconômicas
    pib_crescimento = np.random.normal(2.5, 2.0, n)
    inflacao = np.abs(np.random.normal(4, 3, n))
    divida_pib = np.random.uniform(30, 120, n)
    reservas_pib = np.random.uniform(5, 40, n)
    
    # Rating (escala numérica, com ruído)
    rating_score = (50 
                   + 3 * pib_crescimento 
                   - 2 * inflacao 
                   - 0.3 * divida_pib 
                   + 0.5 * reservas_pib
                   + np.random.normal(0, 5, n))
    
    # Adicionar heterocedasticidade e autocorrelação leves
    rating_score = np.clip(rating_score, 0, 100)
    
    return pd.DataFrame({
        'Rating': rating_score,
        'PIB_Crescimento': pib_crescimento,
        'Inflacao': inflacao,
        'Divida_PIB': divida_pib,
        'Reservas_PIB': reservas_pib
    })


# =============================================================================
# FUNÇÕES AUXILIARES PARA CÁLCULOS E TESTES
# =============================================================================

def fit_ols_closed_form(X: np.ndarray, y: np.ndarray) -> dict:
    """Calcula OLS via fórmula matricial."""
    n, k = X.shape
    
    # Adicionar constante se não existir
    if not np.allclose(X[:, 0], 1):
        X = np.column_stack([np.ones(n), X])
        k = X.shape[1]
    
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ X.T @ y
    
    y_hat = X @ beta_hat
    residuals = y - y_hat
    
    SSE = np.sum(residuals**2)
    SST = np.sum((y - np.mean(y))**2)
    
    r_squared = 1 - SSE / SST
    r_squared_adj = 1 - (SSE / (n - k)) / (SST / (n - 1))
    
    s2 = SSE / (n - k)
    var_beta = s2 * np.diag(XtX_inv)
    se_beta = np.sqrt(var_beta)
    
    t_stats = beta_hat / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
    
    return {
        'beta': beta_hat,
        'se': se_beta,
        't_stats': t_stats,
        'p_values': p_values,
        'y_hat': y_hat,
        'residuals': residuals,
        'SSE': SSE,
        'SST': SST,
        'r_squared': r_squared,
        'r_squared_adj': r_squared_adj,
        's2': s2,
        'n': n,
        'k': k,
        'X': X,
        'XtX_inv': XtX_inv
    }


def white_test(X: np.ndarray, residuals: np.ndarray) -> dict:
    """Teste de White para heterocedasticidade."""
    n = len(residuals)
    u2 = residuals**2
    
    # Regressão auxiliar: u² ~ X, X², X*X (cross products)
    # Simplificado: u² ~ constante, x, x²
    if X.shape[1] >= 2:
        x = X[:, 1]  # Primeira variável explicativa (sem constante)
    else:
        x = X[:, 0]
    
    Z = np.column_stack([np.ones(n), x, x**2])
    
    # OLS da regressão auxiliar
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    gamma = ZtZ_inv @ Z.T @ u2
    u2_hat = Z @ gamma
    
    # R² da regressão auxiliar
    SSR_aux = np.sum((u2_hat - np.mean(u2))**2)
    SST_aux = np.sum((u2 - np.mean(u2))**2)
    r2_aux = SSR_aux / SST_aux if SST_aux > 0 else 0
    
    # Estatística LM = n * R²
    lm_stat = n * r2_aux
    df = Z.shape[1] - 1  # Graus de liberdade
    p_value = 1 - stats.chi2.cdf(lm_stat, df)
    
    return {
        'lm_stat': lm_stat,
        'p_value': p_value,
        'df': df,
        'r2_aux': r2_aux
    }


def durbin_watson(residuals: np.ndarray) -> float:
    """Calcula a estatística de Durbin-Watson."""
    diff = np.diff(residuals)
    dw = np.sum(diff**2) / np.sum(residuals**2)
    return dw


def breusch_godfrey(residuals: np.ndarray, X: np.ndarray, lags: int = 1) -> dict:
    """Teste de Breusch-Godfrey para autocorrelação."""
    n = len(residuals)
    
    # Criar lags dos resíduos
    Z = X.copy()
    for lag in range(1, lags + 1):
        lagged_res = np.zeros(n)
        lagged_res[lag:] = residuals[:-lag]
        Z = np.column_stack([Z, lagged_res])
    
    # Regressão auxiliar: u ~ X, u_{t-1}, ..., u_{t-p}
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    gamma = ZtZ_inv @ Z.T @ residuals
    u_hat = Z @ gamma
    
    # R² da regressão auxiliar
    SSR_aux = np.sum(u_hat**2)
    SST_aux = np.sum(residuals**2)
    r2_aux = SSR_aux / SST_aux if SST_aux > 0 else 0
    
    # Estatística LM = n * R²
    lm_stat = n * r2_aux
    p_value = 1 - stats.chi2.cdf(lm_stat, lags)
    
    return {
        'lm_stat': lm_stat,
        'p_value': p_value,
        'lags': lags,
        'r2_aux': r2_aux
    }


def robust_se(X: np.ndarray, residuals: np.ndarray, XtX_inv: np.ndarray) -> np.ndarray:
    """Calcula erros padrão robustos (HC0 - White)."""
    n, k = X.shape
    
    # Matriz de covariância robusta: (X'X)^{-1} X' diag(u²) X (X'X)^{-1}
    u2 = residuals**2
    meat = X.T @ np.diag(u2) @ X
    var_robust = XtX_inv @ meat @ XtX_inv
    
    return np.sqrt(np.diag(var_robust))


def newey_west_se(X: np.ndarray, residuals: np.ndarray, XtX_inv: np.ndarray, 
                  max_lag: int = None) -> np.ndarray:
    """Calcula erros padrão Newey-West (HAC)."""
    n, k = X.shape
    
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))
    
    # Começar com matriz HC0
    u = residuals
    S = np.zeros((k, k))
    
    for t in range(n):
        S += u[t]**2 * np.outer(X[t], X[t])
    
    # Adicionar termos de autocovariância
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        for t in range(lag, n):
            cross = u[t] * u[t - lag] * (np.outer(X[t], X[t - lag]) + np.outer(X[t - lag], X[t]))
            S += weight * cross
    
    var_nw = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(var_nw))


def compute_vif(X: np.ndarray) -> np.ndarray:
    """Calcula VIF para cada variável (excluindo constante)."""
    n, k = X.shape
    
    # Identificar se primeira coluna é constante
    start_idx = 1 if np.allclose(X[:, 0], 1) else 0
    
    vifs = []
    for j in range(start_idx, k):
        # Regressão de X_j contra as outras variáveis
        mask = [i for i in range(k) if i != j]
        X_others = X[:, mask]
        x_j = X[:, j]
        
        # OLS
        XtX_inv = np.linalg.inv(X_others.T @ X_others)
        beta = XtX_inv @ X_others.T @ x_j
        x_hat = X_others @ beta
        
        # R² e VIF
        ss_res = np.sum((x_j - x_hat)**2)
        ss_tot = np.sum((x_j - np.mean(x_j))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vifs.append(vif)
    
    return np.array(vifs)


def jarque_bera(residuals: np.ndarray) -> dict:
    """Teste de Jarque-Bera para normalidade."""
    n = len(residuals)
    
    # Padronizar resíduos
    u = residuals - np.mean(residuals)
    s = np.std(residuals, ddof=1)
    u_std = u / s
    
    # Skewness e Kurtosis
    skew = np.mean(u_std**3)
    kurt = np.mean(u_std**4)
    
    # Estatística JB
    jb_stat = n * (skew**2 / 6 + (kurt - 3)**2 / 24)
    p_value = 1 - stats.chi2.cdf(jb_stat, 2)
    
    return {
        'jb_stat': jb_stat,
        'p_value': p_value,
        'skewness': skew,
        'kurtosis': kurt
    }


def ramsey_reset(y: np.ndarray, X: np.ndarray, residuals: np.ndarray, 
                 y_hat: np.ndarray, powers: int = 2) -> dict:
    """Teste RESET de Ramsey para forma funcional."""
    n = len(y)
    
    # Adicionar potências de y_hat ao modelo
    Z = X.copy()
    for p in range(2, powers + 2):
        Z = np.column_stack([Z, y_hat**p])
    
    # Modelo expandido
    ols_expanded = fit_ols_closed_form(Z, y)
    
    # Teste F para os termos adicionais
    k_original = X.shape[1]
    k_expanded = Z.shape[1]
    q = k_expanded - k_original
    
    sse_restricted = np.sum(residuals**2)
    sse_unrestricted = ols_expanded['SSE']
    
    f_stat = ((sse_restricted - sse_unrestricted) / q) / (sse_unrestricted / (n - k_expanded))
    p_value = 1 - stats.f.cdf(f_stat, q, n - k_expanded)
    
    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'df1': q,
        'df2': n - k_expanded
    }


def chow_test(y: np.ndarray, X: np.ndarray, break_point: int) -> dict:
    """Teste de Chow para quebra estrutural."""
    n, k = X.shape
    
    # Modelo pooled (todo o período)
    ols_pooled = fit_ols_closed_form(X, y)
    sse_pooled = ols_pooled['SSE']
    
    # Modelo antes da quebra
    ols_before = fit_ols_closed_form(X[:break_point], y[:break_point])
    sse_before = ols_before['SSE']
    
    # Modelo depois da quebra
    ols_after = fit_ols_closed_form(X[break_point:], y[break_point:])
    sse_after = ols_after['SSE']
    
    # Estatística F de Chow
    sse_unrestricted = sse_before + sse_after
    
    f_stat = ((sse_pooled - sse_unrestricted) / k) / (sse_unrestricted / (n - 2 * k))
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)
    
    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'sse_pooled': sse_pooled,
        'sse_before': sse_before,
        'sse_after': sse_after,
        'df1': k,
        'df2': n - 2 * k
    }


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: Por que as suposições importam? (BLUE e risco decisório)"""
    st.header("🎯 Por que as Suposições Importam?")
    
    st.markdown("""
    O OLS é **BLUE** (Best Linear Unbiased Estimator) *somente se* certas suposições forem válidas.
    Quando falham, os coeficientes podem estar ok, mas **erros padrão e testes enganam**.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("As 5 Suposições Clássicas")
        
        suposicoes = [
            ("1️⃣ Linearidade", "Y é função linear de X mais erro", 
             "Coeficientes não capturam a relação verdadeira"),
            ("2️⃣ Exogeneidade", "E(u|X) = 0 — erro não correlacionado com X",
             "Coeficientes viesados e inconsistentes"),
            ("3️⃣ Homocedasticidade", "Var(u|X) = σ² constante",
             "Erros padrão incorretos → testes inválidos"),
            ("4️⃣ Não-autocorrelação", "Cov(uᵢ, uⱼ) = 0 para i ≠ j",
             "Erros padrão subestimados → falsa precisão"),
            ("5️⃣ Normalidade", "u ~ N(0, σ²)",
             "Inferência em amostras pequenas comprometida")
        ]
        
        for titulo, descricao, consequencia in suposicoes:
            with st.expander(titulo):
                st.markdown(f"**O que diz:** {descricao}")
                st.markdown(f"**Se falhar:** {consequencia}")
    
    with col2:
        st.subheader("Resumo Visual: Impacto das Violações")
        
        # Criar tabela resumo
        impact_data = {
            'Violação': ['Heterocedasticidade', 'Autocorrelação', 'Multicolinearidade', 
                        'Não-normalidade', 'Forma funcional errada'],
            'β viesado?': ['Não', 'Não', 'Não', 'Não', 'Sim'],
            'SE incorreto?': ['Sim ⚠️', 'Sim ⚠️', 'Inflado ⚠️', 'Não*', 'Sim'],
            'Testes inválidos?': ['Sim ⚠️', 'Sim ⚠️', 'Parcial', 'Em amostras pequenas', 'Sim ⚠️']
        }
        st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)
        
        st.caption("*Em amostras grandes, normalidade é menos crítica (Teorema Central do Limite)")
        
        st.warning("""
        ⚠️ **Risco decisório:** Você pode concluir que uma variável é significativa 
        quando não é (falso positivo), ou ter excesso de confiança na precisão do modelo.
        """)
    
    st.markdown("---")
    
    st.subheader("📋 Mini-Checklist: Quando Desconfiar do Modelo?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - [ ] Resíduos mostram padrão sistemático (funil, curva)?
        - [ ] Dados são séries temporais (risco de autocorrelação)?
        - [ ] Variáveis explicativas são muito correlacionadas?
        """)
    
    with col2:
        st.markdown("""
        - [ ] Há outliers ou eventos extremos nos dados?
        - [ ] O modelo foi estimado em período diferente do usado?
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Nunca confia cegamente em p-valores sem verificar diagnósticos
    - Exige robustez: "Os resultados mudam com erros padrão robustos?"
    """)


def render_section_S2():
    """S2: Heterocedasticidade (incerteza não constante)"""
    st.header("📊 Heterocedasticidade: Variância Não Constante")
    
    st.markdown("""
    **Heterocedasticidade** ocorre quando a variância do erro muda com X.
    Exemplo: gastos mais altos têm maior variabilidade que gastos baixos.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📈 Visual", "🧪 Teste de White", "🛡️ Erros Robustos"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Controles")
            hetero_intensity = st.slider("Intensidade da heterocedasticidade", 
                                        0.0, 2.0, 0.0, 0.1,
                                        help="0 = homocedasticidade; >0 = variância cresce com x")
            
            st.markdown("""
            **O que observar:**
            - Com intensidade = 0: resíduos têm dispersão constante
            - Com intensidade > 0: forma de "funil" (dispersão cresce)
            """)
        
        with col2:
            df = make_hetero_data(n=200, hetero_intensity=hetero_intensity)
            X = np.column_stack([np.ones(len(df)), df['x'].values])
            ols = fit_ols_closed_form(X, df['y'].values)
            
            # Gráfico de resíduos vs x
            fig = px.scatter(x=df['x'], y=ols['residuals'],
                            labels={'x': 'X', 'y': 'Resíduos'},
                            title="Resíduos vs X (detecte o padrão funil)")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            if hetero_intensity > 0.5:
                st.error("🔍 Padrão de funil visível — heterocedasticidade provável!")
            elif hetero_intensity > 0:
                st.warning("⚠️ Leve padrão de dispersão crescente")
            else:
                st.success("✅ Dispersão aparentemente constante")
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Teste de White")
            
            st.markdown("""
            **Hipóteses:**
            - H₀: Homocedasticidade (variância constante)
            - H₁: Heterocedasticidade
            
            **Método:** Regride u² contra X e X² e testa se coeficientes são significativos.
            """)
            
            df = make_hetero_data(n=200, hetero_intensity=hetero_intensity)
            X = np.column_stack([np.ones(len(df)), df['x'].values])
            ols = fit_ols_closed_form(X, df['y'].values)
            
            white = white_test(X, ols['residuals'])
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Estatística LM", f"{white['lm_stat']:.2f}")
            col_m2.metric("p-valor", f"{white['p_value']:.4f}")
            
            if white['p_value'] < 0.05:
                st.error("❌ Rejeita H₀: Evidência de heterocedasticidade!")
            else:
                st.success("✅ Não rejeita H₀: Sem evidência forte de heterocedasticidade")
        
        with col2:
            st.subheader("Interpretação Gerencial")
            
            st.markdown("""
            **Se detectar heterocedasticidade:**
            
            1. **Coeficientes (β):** Ainda são não-viesados ✓
            2. **Erros padrão:** São incorretos ✗
            3. **Testes t e F:** São inválidos ✗
            4. **Intervalos de confiança:** São incorretos ✗
            
            **Risco:** Você pode pensar que uma variável é significativa quando não é!
            """)
    
    with tab3:
        st.subheader("Solução: Erros Padrão Robustos")
        
        col1, col2 = st.columns([1, 1])
        
        df = make_hetero_data(n=200, hetero_intensity=hetero_intensity)
        X = np.column_stack([np.ones(len(df)), df['x'].values])
        ols = fit_ols_closed_form(X, df['y'].values)
        
        se_classic = ols['se']
        se_robust = robust_se(ols['X'], ols['residuals'], ols['XtX_inv'])
        
        with col1:
            st.markdown("**Comparação de Erros Padrão:**")
            
            comp_df = pd.DataFrame({
                'Variável': ['Intercepto', 'X'],
                'Coeficiente': ols['beta'].round(3),
                'SE Clássico': se_classic.round(4),
                'SE Robusto': se_robust.round(4),
                'Diferença %': ((se_robust / se_classic - 1) * 100).round(1)
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            if hetero_intensity > 0.5:
                st.warning("⚠️ Note como o SE robusto difere do clássico!")
        
        with col2:
            st.markdown("**Mitigações:**")
            
            st.markdown("""
            1. **Erros padrão robustos (HC):** Corrige os SEs sem alterar β
            2. **Transformação log:** Se variância proporcional ao nível
            3. **Weighted Least Squares:** Se conhece a estrutura da variância
            4. **Reespecificação:** Adicionar variáveis omitidas
            """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Sempre visualiza resíduos vs X antes de confiar nos testes
    - Usa erros padrão robustos como padrão em dados cross-section
    """)


def render_section_S3():
    """S3: Autocorrelação (o fantasma do passado)"""
    st.header("📈 Autocorrelação: Erros Correlacionados no Tempo")
    
    st.markdown("""
    **Autocorrelação** ocorre quando o erro de hoje depende do erro de ontem.
    Comum em séries temporais: se subestimamos hoje, provavelmente subestimamos amanhã.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Visual", "🧪 Testes (DW/BG)", "🛡️ Newey-West"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Controles")
            rho = st.slider("ρ (autocorrelação AR(1))", -0.9, 0.9, 0.0, 0.1,
                           help="0 = sem autocorrelação; próximo de ±1 = forte autocorrelação")
            
            st.markdown(f"""
            **Modelo do erro:** uₜ = {rho:.1f} × uₜ₋₁ + eₜ
            
            - ρ = 0: Erros independentes
            - ρ > 0: Autocorrelação positiva (mais comum)
            - ρ < 0: Autocorrelação negativa
            """)
        
        with col2:
            df = make_autocorr_ts_data(n=100, rho=rho)
            X = np.column_stack([np.ones(len(df)), df['x'].values])
            ols = fit_ols_closed_form(X, df['y'].values)
            
            # Gráfico de resíduos ao longo do tempo
            fig = make_subplots(rows=2, cols=1, 
                               subplot_titles=["Resíduos ao Longo do Tempo", "Resíduo t vs Resíduo t-1"])
            
            fig.add_trace(go.Scatter(x=df['t'], y=ols['residuals'], mode='lines+markers',
                                    marker=dict(size=5), name='Resíduos'),
                         row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Scatter de u_t vs u_{t-1}
            fig.add_trace(go.Scatter(x=ols['residuals'][:-1], y=ols['residuals'][1:],
                                    mode='markers', name='u_t vs u_{t-1}'),
                         row=2, col=1)
            
            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(title_text="Tempo", row=1, col=1)
            fig.update_xaxes(title_text="Resíduo t-1", row=2, col=1)
            fig.update_yaxes(title_text="Resíduo t", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
            
            if abs(rho) > 0.5:
                st.error("🔍 Padrão claro de persistência nos resíduos!")
            elif abs(rho) > 0.2:
                st.warning("⚠️ Alguma dependência temporal visível")
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        df = make_autocorr_ts_data(n=100, rho=rho)
        X = np.column_stack([np.ones(len(df)), df['x'].values])
        ols = fit_ols_closed_form(X, df['y'].values)
        
        with col1:
            st.subheader("Teste Durbin-Watson")
            
            dw = durbin_watson(ols['residuals'])
            
            st.metric("Estatística DW", f"{dw:.2f}")
            
            st.markdown("""
            **Interpretação:**
            - DW ≈ 2: Sem autocorrelação
            - DW < 2: Autocorrelação positiva
            - DW > 2: Autocorrelação negativa
            
            **Regra prática:** DW < 1.5 ou DW > 2.5 → suspeitar
            """)
            
            if dw < 1.5:
                st.error("⚠️ DW baixo: provável autocorrelação positiva")
            elif dw > 2.5:
                st.warning("⚠️ DW alto: possível autocorrelação negativa")
            else:
                st.success("✅ DW próximo de 2: sem evidência forte")
        
        with col2:
            st.subheader("Teste Breusch-Godfrey")
            
            bg = breusch_godfrey(ols['residuals'], ols['X'], lags=1)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Estatística LM", f"{bg['lm_stat']:.2f}")
            col_m2.metric("p-valor", f"{bg['p_value']:.4f}")
            
            st.markdown("""
            **Vantagem sobre DW:**
            - Funciona com variáveis defasadas no modelo
            - Testa múltiplos lags
            - Fornece p-valor direto
            """)
            
            if bg['p_value'] < 0.05:
                st.error("❌ Rejeita H₀: Evidência de autocorrelação!")
            else:
                st.success("✅ Não rejeita H₀")
    
    with tab3:
        st.subheader("Solução: Erros Padrão Newey-West (HAC)")
        
        col1, col2 = st.columns([1, 1])
        
        df = make_autocorr_ts_data(n=100, rho=rho)
        X = np.column_stack([np.ones(len(df)), df['x'].values])
        ols = fit_ols_closed_form(X, df['y'].values)
        
        se_classic = ols['se']
        se_nw = newey_west_se(ols['X'], ols['residuals'], ols['XtX_inv'])
        
        with col1:
            st.markdown("**Comparação de Erros Padrão:**")
            
            comp_df = pd.DataFrame({
                'Variável': ['Intercepto', 'X'],
                'Coeficiente': ols['beta'].round(3),
                'SE Clássico': se_classic.round(4),
                'SE Newey-West': se_nw.round(4),
                'Razão NW/Clássico': (se_nw / se_classic).round(2)
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            if abs(rho) > 0.3 and np.mean(se_nw / se_classic) > 1.2:
                st.warning("⚠️ SE Newey-West é maior — autocorrelação infla a falsa precisão!")
        
        with col2:
            st.markdown("**Opções de Correção:**")
            
            st.markdown("""
            1. **Newey-West (HAC):** Corrige SEs para heterocedasticidade E autocorrelação
            
            2. **Modelo Dinâmico:** Incluir variável dependente defasada:
               - yₜ = α + βxₜ + γyₜ₋₁ + εₜ
            
            3. **Diferenciação:** Usar Δy = yₜ - yₜ₋₁ como dependente
            
            4. **GLS (Cochrane-Orcutt):** Transformar o modelo
            """)
        
        st.info("""
        💡 **R² inflado:** Com autocorrelação, o R² pode parecer alto porque o modelo 
        "segue" a tendência dos erros, não porque explica bem Y.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Em séries temporais, sempre verifica DW e usa Newey-West por padrão
    - Questiona: "O modelo está prevendo bem ou apenas seguindo a tendência?"
    """)


def render_section_S4():
    """S4: Multicolinearidade (variáveis que dizem a mesma coisa)"""
    st.header("🔗 Multicolinearidade: Variáveis Redundantes")
    
    st.markdown("""
    **Multicolinearidade** ocorre quando variáveis explicativas são altamente correlacionadas.
    O modelo não consegue separar seus efeitos individuais.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Controles")
        
        corr_level = st.slider("Correlação entre X₁ e X₂", 0.0, 0.99, 0.0, 0.05)
        
        st.markdown(f"""
        **Situação:** X₁ e X₂ têm correlação = {corr_level:.2f}
        
        - corr = 0: Variáveis independentes
        - corr > 0.7: Multicolinearidade moderada
        - corr > 0.9: Multicolinearidade severa
        """)
    
    with col2:
        df = make_collinear_data(n=200, corr=corr_level)
        
        # Matriz de correlação
        corr_matrix = df[['x1', 'x2']].corr()
        
        fig = px.imshow(corr_matrix, text_auto='.2f', 
                       color_continuous_scale='RdBu_r',
                       title="Matriz de Correlação")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # VIF e resultados
    st.subheader("Diagnóstico: VIF (Variance Inflation Factor)")
    
    X = np.column_stack([np.ones(len(df)), df['x1'].values, df['x2'].values])
    ols = fit_ols_closed_form(X, df['y'].values)
    vifs = compute_vif(X)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**VIF por Variável:**")
        
        vif_df = pd.DataFrame({
            'Variável': ['X₁', 'X₂'],
            'VIF': vifs.round(2),
            'Status': ['⚠️ Alto' if v > 10 else ('⚡ Moderado' if v > 5 else '✅ OK') for v in vifs]
        })
        st.dataframe(vif_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Regras de bolso:**
        - VIF < 5: Geralmente aceitável
        - VIF 5-10: Preocupante
        - VIF > 10: Multicolinearidade severa
        """)
        
        if max(vifs) > 10:
            st.error("🚨 Multicolinearidade severa detectada!")
        elif max(vifs) > 5:
            st.warning("⚠️ Multicolinearidade moderada")
        else:
            st.success("✅ Sem multicolinearidade problemática")
    
    with col2:
        st.markdown("**Resultados da Regressão:**")
        
        results_df = pd.DataFrame({
            'Variável': ['Intercepto', 'X₁', 'X₂'],
            'β': ols['beta'].round(3),
            'SE': ols['se'].round(3),
            't-stat': ols['t_stats'].round(2),
            'p-valor': ols['p_values'].round(4)
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Efeitos da multicolinearidade:**
        - β's continuam não-viesados, mas...
        - Erros padrão ficam **inflados**
        - Coeficientes ficam **instáveis** (sensíveis a pequenas mudanças)
        - Variáveis podem parecer não-significativas mesmo sendo importantes
        """)
    
    with st.expander("💡 Soluções para Multicolinearidade"):
        st.markdown("""
        **1. Remover redundância:**
        - Excluir uma das variáveis correlacionadas
        - Escolher baseado em teoria ou relevância prática
        
        **2. Criar índices/razões:**
        - Combinar variáveis em um único indicador
        - Ex.: em vez de Receita e Custos, usar Margem
        
        **3. Aumentar amostra:**
        - Mais dados ajudam a separar efeitos (mas nem sempre viável)
        
        **4. Regularização (avançado):**
        - Ridge regression, LASSO
        - Penaliza coeficientes grandes
        
        **5. Aceitar e interpretar com cuidado:**
        - Se o objetivo é previsão (não interpretação), pode ser ok
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Verifica VIF antes de interpretar coeficientes individuais
    - Questiona: "Essas variáveis medem a mesma coisa de formas diferentes?"
    """)


def render_section_S5():
    """S5: Normalidade e Forma Funcional"""
    st.header("📐 Normalidade e Forma Funcional")
    
    tab1, tab2 = st.tabs(["🔔 Normalidade (Jarque-Bera)", "📈 Forma Funcional (RESET)"])
    
    with tab1:
        st.subheader("Teste de Normalidade dos Resíduos")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            outlier_pct = st.slider("% de Outliers", 0.0, 10.0, 0.0, 0.5)
            
            st.markdown("""
            **Por que normalidade importa?**
            - Para testes t e F em amostras pequenas
            - Para intervalos de confiança exatos
            
            **Em amostras grandes:** Menos crítico (Teorema Central do Limite)
            """)
            
            df = make_nonnormal_data(n=200, outlier_pct=outlier_pct)
            X = np.column_stack([np.ones(len(df)), df['x'].values])
            ols = fit_ols_closed_form(X, df['y'].values)
            
            jb = jarque_bera(ols['residuals'])
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Estatística JB", f"{jb['jb_stat']:.2f}")
            col_m2.metric("p-valor", f"{jb['p_value']:.4f}")
            
            col_m3, col_m4 = st.columns(2)
            col_m3.metric("Assimetria", f"{jb['skewness']:.2f}", 
                         help="0 = simétrico")
            col_m4.metric("Curtose", f"{jb['kurtosis']:.2f}",
                         help="3 = normal")
            
            if jb['p_value'] < 0.05:
                st.error("❌ Rejeita normalidade — resíduos não são normais")
            else:
                st.success("✅ Não rejeita normalidade")
        
        with col2:
            # Histograma dos resíduos
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ols['residuals'], nbinsx=30, 
                                       name='Resíduos', opacity=0.7))
            
            # Sobrepor curva normal teórica
            x_norm = np.linspace(ols['residuals'].min(), ols['residuals'].max(), 100)
            y_norm = stats.norm.pdf(x_norm, 0, np.std(ols['residuals'])) * len(ols['residuals']) * (ols['residuals'].max() - ols['residuals'].min()) / 30
            fig.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines',
                                    line=dict(color='red', width=2),
                                    name='Normal teórica'))
            
            fig.update_layout(
                title="Histograma dos Resíduos vs Normal",
                xaxis_title="Resíduos",
                yaxis_title="Frequência",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Teste RESET de Ramsey: A Forma Funcional Está Correta?")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **O que o RESET testa:**
            - H₀: Forma funcional linear é adequada
            - H₁: Termos não-lineares são necessários
            
            **Método:** Adiciona ŷ², ŷ³ ao modelo e testa se são significativos.
            Se forem, a relação linear original está mal especificada.
            """)
            
            usar_quadratico = st.checkbox("Simular relação quadrática verdadeira", value=False)
            
            # Gerar dados
            np.random.seed(42)
            n = 200
            x = np.random.uniform(0, 10, n)
            if usar_quadratico:
                y = 5 + 2*x - 0.2*x**2 + np.random.normal(0, 2, n)
            else:
                y = 5 + 2*x + np.random.normal(0, 2, n)
            
            X = np.column_stack([np.ones(n), x])
            ols = fit_ols_closed_form(X, y)
            
            reset = ramsey_reset(y, ols['X'], ols['residuals'], ols['y_hat'])
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Estatística F", f"{reset['f_stat']:.2f}")
            col_m2.metric("p-valor", f"{reset['p_value']:.4f}")
            
            if reset['p_value'] < 0.05:
                st.error("❌ Rejeita H₀: Forma funcional inadequada! Considere termos não-lineares.")
            else:
                st.success("✅ Não rejeita H₀: Forma linear parece adequada")
        
        with col2:
            # Gráfico
            fig = px.scatter(x=x, y=y, opacity=0.6,
                            labels={'x': 'X', 'y': 'Y'},
                            title="Dados e Ajuste Linear")
            
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = ols['beta'][0] + ols['beta'][1] * x_line
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                    line=dict(color='red'),
                                    name='Ajuste Linear'))
            
            if usar_quadratico:
                # Mostrar ajuste quadrático também
                X_quad = np.column_stack([np.ones(n), x, x**2])
                ols_quad = fit_ols_closed_form(X_quad, y)
                y_quad = ols_quad['beta'][0] + ols_quad['beta'][1]*x_line + ols_quad['beta'][2]*x_line**2
                fig.add_trace(go.Scatter(x=x_line, y=y_quad, mode='lines',
                                        line=dict(color='green', dash='dash'),
                                        name='Ajuste Quadrático'))
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("💡 Recomendações Práticas"):
        st.markdown("""
        **Se normalidade falhar:**
        - Em amostras grandes (n > 100): geralmente não é crítico
        - Verifique outliers e considere removê-los ou usar dummies
        - Bootstrap pode dar inferência mais robusta
        
        **Se forma funcional falhar (RESET):**
        - Adicionar termos quadráticos: x²
        - Usar transformação log: log(x), log(y)
        - Adicionar interações: x₁ × x₂
        - Incluir dummies para regimes (crise/normal)
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Verifica RESET para saber se a relação é realmente linear
    - Não descarta modelo só por falhar JB em amostras grandes
    """)


def render_section_S6():
    """S6: Estabilidade do Modelo e Filosofia de Construção"""
    st.header("🔄 Estabilidade e Construção de Modelos")
    
    tab1, tab2 = st.tabs(["📊 Teste de Chow", "🔧 Geral-para-Específico"])
    
    with tab1:
        st.subheader("Teste de Chow: O Modelo é Estável no Tempo?")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Pergunta central:** Os coeficientes são os mesmos antes e depois de um evento?
            
            Exemplos:
            - Crise de 2008 mudou a relação risco-retorno?
            - Nova regulação alterou o comportamento do mercado?
            - O modelo dos anos 2000 funciona em 2020?
            """)
            
            has_break = st.checkbox("Simular quebra estrutural", value=False)
            break_point = st.slider("Ponto de quebra (observação)", 20, 80, 50)
            
            df = make_structural_break_data(n=100, break_point=break_point, has_break=has_break)
            X = np.column_stack([np.ones(len(df)), df['x'].values])
            
            chow = chow_test(df['y'].values, X, break_point)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Estatística F", f"{chow['f_stat']:.2f}")
            col_m2.metric("p-valor", f"{chow['p_value']:.4f}")
            
            if chow['p_value'] < 0.05:
                st.error("❌ Rejeita estabilidade: Coeficientes mudaram após a quebra!")
            else:
                st.success("✅ Não rejeita estabilidade: Modelo parece consistente")
        
        with col2:
            # Gráfico com cores por regime
            fig = px.scatter(df, x='x', y='y', color='regime',
                            title="Dados por Regime (Antes/Depois da Quebra)")
            
            # Ajustar modelos separados
            X_before = X[:break_point]
            X_after = X[break_point:]
            ols_before = fit_ols_closed_form(X_before, df['y'].values[:break_point])
            ols_after = fit_ols_closed_form(X_after, df['y'].values[break_point:])
            
            x_range = np.linspace(df['x'].min(), df['x'].max(), 50)
            
            fig.add_trace(go.Scatter(
                x=x_range, y=ols_before['beta'][0] + ols_before['beta'][1] * x_range,
                mode='lines', line=dict(color='blue', dash='dash'),
                name=f"Antes: β={ols_before['beta'][1]:.2f}"
            ))
            
            fig.add_trace(go.Scatter(
                x=x_range, y=ols_after['beta'][0] + ols_after['beta'][1] * x_range,
                mode='lines', line=dict(color='red', dash='dash'),
                name=f"Depois: β={ols_after['beta'][1]:.2f}"
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Filosofia Geral-para-Específico")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Abordagem recomendada para construir modelos:**
            
            1. **Começar amplo:** Incluir todas as variáveis teoricamente relevantes
            
            2. **Diagnosticar:** Verificar hetero, auto, colinearidade, forma funcional
            
            3. **Simplificar:** Remover variáveis não significativas (uma por vez)
            
            4. **Validar:** Re-testar diagnósticos após cada mudança
            
            5. **Documentar:** Justificar exclusões e inclusões
            """)
        
        with col2:
            # Fluxograma simplificado
            st.markdown("""
            ```
            ┌─────────────────────┐
            │  Modelo Geral       │
            │  (todas variáveis)  │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Diagnósticos       │
            │  • Hetero (White)   │
            │  • Auto (DW/BG)     │
            │  • Multicolinear    │
            │  • RESET, JB        │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Correções          │
            │  • Erros robustos   │
            │  • Transformações   │
            │  • Remover vars     │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Modelo Final       │
            │  Parcimônia + Fit   │
            └─────────────────────┘
            ```
            """)
        
        st.info("""
        💡 **Princípio:** Prefira modelos mais simples que passam nos diagnósticos 
        a modelos complexos que "encaixam" melhor mas violam suposições.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Testa estabilidade antes de usar modelo histórico para decisões futuras
    - Documenta o processo de construção do modelo para auditoria
    """)


def render_section_S7():
    """S7: Estudo de Caso: Ratings de Crédito Soberano"""
    st.header("🏛️ Estudo de Caso: Determinantes de Rating Soberano")
    
    st.markdown("""
    Vamos aplicar o workflow completo de diagnóstico em um modelo de rating de crédito soberano.
    
    **Variáveis:**
    - **Rating:** Score numérico (0-100)
    - **PIB_Crescimento:** Crescimento do PIB (%)
    - **Inflação:** Taxa de inflação (%)
    - **Dívida_PIB:** Dívida pública / PIB (%)
    - **Reservas_PIB:** Reservas internacionais / PIB (%)
    """)
    
    # Gerar dados
    df = make_ratings_case_data(n=50)
    
    # Mostrar dados
    with st.expander("📊 Ver Dados"):
        st.dataframe(df.round(2), use_container_width=True)
    
    st.subheader("Passo 1: Estimação Inicial")
    
    X = np.column_stack([
        np.ones(len(df)),
        df['PIB_Crescimento'].values,
        df['Inflacao'].values,
        df['Divida_PIB'].values,
        df['Reservas_PIB'].values
    ])
    y = df['Rating'].values
    
    ols = fit_ols_closed_form(X, y)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Resultados da Regressão:**")
        
        nomes = ['Intercepto', 'PIB Crescimento', 'Inflação', 'Dívida/PIB', 'Reservas/PIB']
        results_df = pd.DataFrame({
            'Variável': nomes,
            'Coeficiente': ols['beta'].round(3),
            'SE Clássico': ols['se'].round(3),
            'p-valor': ols['p_values'].round(4)
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    with col2:
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("R²", f"{ols['r_squared']:.3f}")
        col_m2.metric("R² Ajustado", f"{ols['r_squared_adj']:.3f}")
    
    st.subheader("Passo 2: Diagnósticos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Heterocedasticidade
    white = white_test(ols['X'], ols['residuals'])
    with col1:
        st.markdown("**Hetero (White)**")
        st.metric("p-valor", f"{white['p_value']:.3f}")
        if white['p_value'] < 0.05:
            st.error("⚠️ Detectada")
        else:
            st.success("✅ OK")
    
    # Normalidade
    jb = jarque_bera(ols['residuals'])
    with col2:
        st.markdown("**Normal (JB)**")
        st.metric("p-valor", f"{jb['p_value']:.3f}")
        if jb['p_value'] < 0.05:
            st.warning("⚠️ Rejeita")
        else:
            st.success("✅ OK")
    
    # Forma funcional
    reset = ramsey_reset(y, ols['X'], ols['residuals'], ols['y_hat'])
    with col3:
        st.markdown("**RESET**")
        st.metric("p-valor", f"{reset['p_value']:.3f}")
        if reset['p_value'] < 0.05:
            st.error("⚠️ Rejeita")
        else:
            st.success("✅ OK")
    
    # Multicolinearidade
    vifs = compute_vif(ols['X'])
    with col4:
        st.markdown("**VIF máximo**")
        st.metric("VIF", f"{max(vifs):.1f}")
        if max(vifs) > 10:
            st.error("⚠️ Alto")
        elif max(vifs) > 5:
            st.warning("⚠️ Moderado")
        else:
            st.success("✅ OK")
    
    st.subheader("Passo 3: Comparação com Erros Robustos")
    
    se_robust = robust_se(ols['X'], ols['residuals'], ols['XtX_inv'])
    
    comp_df = pd.DataFrame({
        'Variável': nomes,
        'Coeficiente': ols['beta'].round(3),
        'SE Clássico': ols['se'].round(3),
        'SE Robusto': se_robust.round(3),
        't Clássico': (ols['beta'] / ols['se']).round(2),
        't Robusto': (ols['beta'] / se_robust).round(2)
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.subheader("Passo 4: Resumo Executivo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ✅ Conclusões do Modelo")
        st.markdown(f"""
        **Variáveis significativas (SE robusto):**
        """)
        for i, nome in enumerate(nomes[1:], 1):
            t_rob = abs(ols['beta'][i] / se_robust[i])
            sig = "✓" if t_rob > 1.96 else "✗"
            direcao = "↑" if ols['beta'][i] > 0 else "↓"
            st.markdown(f"- {nome}: {sig} (β={ols['beta'][i]:.2f}, {direcao} rating)")
    
    with col2:
        st.markdown("### ⚠️ Riscos e Limitações")
        st.markdown("""
        - Amostra pequena (n=50) limita inferência
        - Possível endogeneidade: rating afeta economia?
        - Modelo simplificado: fatores políticos não incluídos
        - Estabilidade não testada: modelo pode não valer em crises
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa o modelo para entender drivers, não para previsão precisa
    - Reporta resultados com erros robustos e documenta limitações
    """)


def render_section_S8():
    """S8: Resumo Executivo e Ponte para o Próximo Módulo"""
    st.header("📋 Resumo Executivo: Diagnósticos do CLRM")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Principais Falhas e Como Lidar
        
        | Problema | Detecção | Consequência | Solução |
        |----------|----------|--------------|---------|
        | **Heterocedasticidade** | Visual (funil), White | SE incorretos | Erros robustos |
        | **Autocorrelação** | DW, Breusch-Godfrey | SE subestimados | Newey-West, dinâmicos |
        | **Multicolinearidade** | Correlação, VIF | SE inflados | Remover, combinar vars |
        | **Não-normalidade** | JB, histograma | Inferência (n pequeno) | Amostra maior, bootstrap |
        | **Forma funcional** | RESET | β viesado | Adicionar não-lineares |
        | **Instabilidade** | Chow | Modelo obsoleto | Re-estimar, dummies |
        
        ### Workflow Recomendado
        
        1. **Estimar** modelo inicial com todas variáveis relevantes
        2. **Visualizar** resíduos vs X e vs tempo
        3. **Testar** hetero (White), auto (DW/BG), forma (RESET), normalidade (JB)
        4. **Verificar** multicolinearidade (VIF) e estabilidade (Chow se aplicável)
        5. **Corrigir** usando erros robustos ou reespecificação
        6. **Comparar** resultados antes/depois das correções
        7. **Documentar** limitações e riscos
        """)
    
    with col2:
        st.markdown("### 🧪 Quiz Final")
        
        st.markdown("""
        Um analista encontrou:
        - p-valor White = 0.02
        - VIF máximo = 3.2
        - p-valor JB = 0.15
        - DW = 1.95
        """)
        
        resposta = st.radio(
            "Qual o principal problema?",
            ["Multicolinearidade", "Heterocedasticidade", 
             "Não-normalidade", "Autocorrelação"],
            key="quiz_final"
        )
        
        if st.button("Verificar", key="btn_final"):
            if resposta == "Heterocedasticidade":
                st.success("""
                ✅ **Correto!** 
                - White p=0.02 < 0.05 → rejeita homocedasticidade
                - VIF=3.2 < 5 → OK
                - JB p=0.15 > 0.05 → não rejeita normalidade
                - DW≈2 → sem autocorrelação
                
                **Ação:** Usar erros padrão robustos (HC).
                """)
            else:
                st.error("O problema é **heterocedasticidade** (White p=0.02)")
    
    st.markdown("---")
    
    st.subheader("🔜 Próximo Módulo: Séries Temporais")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Processos ARIMA:**
        - AR: Autoregressivo
        - I: Integrado
        - MA: Média Móvel
        """)
    
    with col2:
        st.markdown("""
        **Estacionariedade:**
        - Testes de raiz unitária
        - Diferenciação
        - Tendências
        """)
    
    with col3:
        st.markdown("""
        **Forecasting:**
        - Previsão pontual
        - Intervalos de previsão
        - Avaliação de acurácia
        """)
    
    st.success("""
    🎓 **Mensagem final:** Diagnósticos não são formalidade — são a diferença entre 
    uma análise confiável e uma ilusão estatística. Sempre verifique antes de decidir.
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Exige diagnósticos em toda análise quantitativa apresentada
    - Questiona: "Esses resultados são robustos a diferentes especificações?"
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("🔍 Módulo 4: Diagnósticos do CLRM")
    st.markdown("**Laboratório de Econometria** | Suposições, Testes e Correções")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Explicar por que as **suposições do CLRM** são críticas para decisões
        - Detectar e mitigar **heterocedasticidade** (White, erros robustos)
        - Detectar e mitigar **autocorrelação** (DW, Breusch-Godfrey, Newey-West)
        - Diagnosticar **multicolinearidade** (VIF) e discutir soluções
        - Avaliar **normalidade** (Jarque-Bera) e **forma funcional** (RESET)
        - Testar **estabilidade** (Chow) e aplicar filosofia geral-para-específico
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "🎯 Por que Suposições Importam",
        "S2": "📊 Heterocedasticidade",
        "S3": "📈 Autocorrelação",
        "S4": "🔗 Multicolinearidade",
        "S5": "📐 Normalidade e Forma Funcional",
        "S6": "🔄 Estabilidade e Construção",
        "S7": "🏛️ Caso: Ratings Soberanos",
        "S8": "📋 Resumo e Próximos Passos"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Dica:** Diagnósticos são essenciais 
    para confiar nos resultados do modelo.
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
            page_title="Módulo 4: Diagnósticos do CLRM",
            page_icon="🔍",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()