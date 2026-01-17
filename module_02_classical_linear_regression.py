"""
Laboratório de Econometria - Module 2: Classical Linear Regression (CLRM)
Aplicativo educacional interativo para regressão linear aplicada a negócios.
Público-alvo: alunos de MBA com perfis quantitativos heterogêneos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# FUNÇÕES AUXILIARES PARA GERAÇÃO DE DADOS E CÁLCULOS
# =============================================================================

@st.cache_data
def make_regression_data(n: int = 100, alpha: float = 10.0, beta: float = 2.0, 
                         sigma: float = 5.0, seed: int = 42) -> pd.DataFrame:
    """Gera dados sintéticos para regressão simples."""
    np.random.seed(seed)
    x = np.random.uniform(10, 50, n)
    u = np.random.normal(0, sigma, n)
    y = alpha + beta * x + u
    return pd.DataFrame({'x': x, 'y': y, 'u': u})


def fit_ols_closed_form(x: np.ndarray, y: np.ndarray) -> dict:
    """Calcula OLS via fórmula fechada (sem statsmodels)."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Coeficientes
    beta_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    alpha_hat = y_mean - beta_hat * x_mean
    
    # Valores ajustados e resíduos
    y_hat = alpha_hat + beta_hat * x
    residuals = y - y_hat
    
    # Soma dos quadrados
    SSE = np.sum(residuals**2)  # Sum of Squared Errors
    SST = np.sum((y - y_mean)**2)  # Total Sum of Squares
    SSR = SST - SSE  # Regression Sum of Squares
    
    # R²
    r_squared = 1 - (SSE / SST) if SST > 0 else 0
    
    # Erro padrão dos resíduos
    s2 = SSE / (n - 2)  # Variância dos resíduos
    se_residuals = np.sqrt(s2)
    
    # Erros padrão dos coeficientes
    se_beta = np.sqrt(s2 / np.sum((x - x_mean)**2))
    se_alpha = np.sqrt(s2 * (1/n + x_mean**2 / np.sum((x - x_mean)**2)))
    
    # Estatísticas t
    t_beta = beta_hat / se_beta
    t_alpha = alpha_hat / se_alpha
    
    # P-valores (aproximação usando distribuição normal para n grande)
    from scipy import stats
    p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), n - 2))
    p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), n - 2))
    
    # Intervalos de confiança (95%)
    t_crit = stats.t.ppf(0.975, n - 2)
    ci_beta = (beta_hat - t_crit * se_beta, beta_hat + t_crit * se_beta)
    ci_alpha = (alpha_hat - t_crit * se_alpha, alpha_hat + t_crit * se_alpha)
    
    return {
        'alpha': alpha_hat,
        'beta': beta_hat,
        'y_hat': y_hat,
        'residuals': residuals,
        'SSE': SSE,
        'SST': SST,
        'SSR': SSR,
        'r_squared': r_squared,
        'se_residuals': se_residuals,
        'se_alpha': se_alpha,
        'se_beta': se_beta,
        't_alpha': t_alpha,
        't_beta': t_beta,
        'p_alpha': p_alpha,
        'p_beta': p_beta,
        'ci_alpha': ci_alpha,
        'ci_beta': ci_beta,
        'n': n
    }


@st.cache_data
def simulate_capm_data(n: int = 60, beta_true: float = 1.2, alpha_true: float = 0.0,
                       sigma: float = 2.0, rf: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Simula dados de retorno de fundo vs mercado (CAPM)."""
    np.random.seed(seed)
    # Retorno do mercado (em % mensal)
    rm = np.random.normal(1.0, 4.0, n)  # Média 1%, vol 4% ao mês
    # Prêmio de risco do mercado
    rm_rf = rm - rf
    # Retorno do fundo
    rf_fund = rf + alpha_true + beta_true * rm_rf + np.random.normal(0, sigma, n)
    return pd.DataFrame({
        'Retorno_Mercado': rm,
        'Retorno_Fundo': rf_fund,
        'Premio_Mercado': rm_rf,
        'Excesso_Fundo': rf_fund - rf
    })


@st.cache_data
def simulate_jensen_alpha(n: int = 60, alpha_true: float = 0.5, beta_true: float = 1.0,
                          sigma: float = 1.5, seed: int = 42) -> pd.DataFrame:
    """Simula dados para análise de Alfa de Jensen."""
    np.random.seed(seed)
    rf = 0.4  # Taxa livre de risco mensal
    rm = np.random.normal(1.2, 4.0, n)
    rm_rf = rm - rf
    # Excesso de retorno do fundo
    ri_rf = alpha_true + beta_true * rm_rf + np.random.normal(0, sigma, n)
    return pd.DataFrame({
        'Excesso_Mercado': rm_rf,
        'Excesso_Fundo': ri_rf,
        'Retorno_Mercado': rm,
        'Retorno_Fundo': ri_rf + rf
    })


def make_endogenous_data(n: int = 100, alpha: float = 10.0, beta_true: float = 2.0,
                         sigma: float = 5.0, corr_ux: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Gera dados com possível violação de exogeneidade (correlação entre u e x)."""
    np.random.seed(seed)
    
    # Variável omitida z que afeta tanto x quanto y
    z = np.random.normal(0, 1, n)
    
    # x é parcialmente determinado por z
    x = 30 + 5 * corr_ux * z + np.random.normal(0, 5, n)
    
    # Erro u também é afetado por z (criando endogeneidade)
    u = sigma * corr_ux * z + np.random.normal(0, sigma * (1 - abs(corr_ux)), n)
    
    # y depende de x e u
    y = alpha + beta_true * x + u
    
    return pd.DataFrame({'x': x, 'y': y, 'u': u, 'z': z})


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: Introdução e Motivação de Negócios"""
    st.header("📈 Introdução: Por que Regressão?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        A **regressão** é a ferramenta mais usada em análise quantitativa de negócios.
        Ela responde à pergunta central: *"Qual o efeito de X sobre Y?"*
        
        Diferente da correlação (que mede associação), a regressão **modela a dependência**
        de uma variável (Y) em relação a outra (X).
        """)
        
        caso = st.selectbox(
            "Selecione um caso de negócio:",
            ["Marketing → Vendas", "Preço → Demanda", 
             "Taxa de Juros → Inadimplência", "Mercado → Retorno do Fundo"]
        )
        
        casos_config = {
            "Marketing → Vendas": {"x_label": "Investimento em Marketing (R$ mil)", 
                                   "y_label": "Vendas (unidades)", "alpha": 100, "beta": 5, "sigma": 50},
            "Preço → Demanda": {"x_label": "Preço (R$)", 
                                "y_label": "Demanda (unidades)", "alpha": 500, "beta": -8, "sigma": 30},
            "Taxa de Juros → Inadimplência": {"x_label": "Taxa de Juros (%)", 
                                               "y_label": "Taxa de Inadimplência (%)", "alpha": 2, "beta": 0.5, "sigma": 1},
            "Mercado → Retorno do Fundo": {"x_label": "Retorno do Mercado (%)", 
                                           "y_label": "Retorno do Fundo (%)", "alpha": 0.5, "beta": 1.2, "sigma": 2}
        }
        
        config = casos_config[caso]
        
        st.info(f"""
        **Pergunta de negócio:** Se aumentarmos {config['x_label'].split('(')[0].strip().lower()}, 
        qual o impacto esperado em {config['y_label'].split('(')[0].strip().lower()}?
        """)
    
    with col2:
        # Gerar dados do caso selecionado
        np.random.seed(42)
        n = 50
        x = np.random.uniform(10, 50, n)
        y = config['alpha'] + config['beta'] * x + np.random.normal(0, config['sigma'], n)
        
        df = pd.DataFrame({'x': x, 'y': y})
        corr = np.corrcoef(x, y)[0, 1]
        
        fig = px.scatter(df, x='x', y='y', 
                        labels={'x': config['x_label'], 'y': config['y_label']},
                        title="Nuvem de Dados: Visualize Antes de Modelar")
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Correlação", f"{corr:.2f}")
    
    with st.expander("📊 Regressão vs Correlação: Qual a Diferença?"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Correlação:**
            - Mede associação simétrica entre X e Y
            - Não distingue causa de efeito
            - Varia de -1 a +1
            - Pergunta: *"X e Y andam juntos?"*
            """)
        with col_b:
            st.markdown("""
            **Regressão:**
            - Modela Y como função de X (assimétrica)
            - Y é variável dependente (aleatória)
            - X é variável explicativa (pode ser fixa)
            - Pergunta: *"Quanto Y muda se X variar?"*
            """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa regressão para **quantificar** o efeito de decisões (investimento, preço, etc.)
    - Visualiza os dados antes de confiar em qualquer modelo
    """)


def render_section_S2():
    """S2: O Modelo de Regressão Simples"""
    st.header("📐 O Modelo de Regressão Simples")
    
    st.markdown("""
    O modelo básico de regressão é:
    
    $$y = \\alpha + \\beta x + u$$
    
    Em linguagem de negócios:
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        | Componente | Significado | Exemplo |
        |------------|-------------|---------|
        | **y** | Variável objetivo (o que queremos explicar/prever) | Vendas, Retorno, Demanda |
        | **x** | Driver / variável explicativa | Marketing, Preço, Mercado |
        | **α (alfa)** | Intercepto (valor base de y quando x=0) | Vendas "orgânicas" |
        | **β (beta)** | Efeito marginal de x sobre y | Impacto de +1 em marketing |
        | **u** | Termo de erro (tudo que não observamos) | Fatores não modelados |
        """)
        
        st.subheader("O que é o termo de erro (u)?")
        
        erro_tipo = st.radio(
            "O erro captura:",
            ["Variáveis omitidas", "Erro de medição", "Aleatoriedade comportamental"],
            horizontal=True
        )
        
        erro_explicacao = {
            "Variáveis omitidas": "Fatores que afetam Y mas não estão no modelo (ex.: qualidade do produto, sazonalidade).",
            "Erro de medição": "Imprecisão nos dados coletados (ex.: vendas estimadas, não exatas).",
            "Aleatoriedade comportamental": "Variação natural no comportamento de consumidores ou mercados."
        }
        
        st.info(f"💡 **{erro_tipo}:** {erro_explicacao[erro_tipo]}")
    
    with col2:
        st.subheader("Gerador de Dados Interativo")
        
        alpha = st.slider("α (intercepto)", -20.0, 50.0, 10.0, 1.0)
        beta = st.slider("β (efeito de x)", -5.0, 5.0, 2.0, 0.1)
        sigma = st.slider("σ (nível de ruído)", 1.0, 30.0, 10.0, 1.0)
        n = st.slider("n (tamanho da amostra)", 20, 200, 50, 10)
        
        df = make_regression_data(n=n, alpha=alpha, beta=beta, sigma=sigma)
        
        fig = px.scatter(df, x='x', y='y', opacity=0.7,
                        title=f"y = {alpha:.1f} + {beta:.1f}x + erro")
        
        # Adicionar reta verdadeira
        x_line = np.array([df['x'].min(), df['x'].max()])
        y_line = alpha + beta * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='Relação Verdadeira'))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📖 Nota Técnica: Por que 'erro' e não 'resíduo'?"):
        st.markdown("""
        - **Erro (u):** É o termo teórico, não observável. Representa tudo que afeta Y além de X.
        - **Resíduo (û):** É a estimativa do erro, calculada após ajustar a regressão: û = y - ŷ.
        
        Na prática, trabalhamos com resíduos porque o erro verdadeiro é desconhecido.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Entende que o modelo é uma **simplificação** — o erro sempre existe
    - Questiona: "Quais variáveis importantes podem estar no termo de erro?"
    """)


def render_section_S3():
    """S3: A Reta de Melhor Ajuste (OLS) — Intuição"""
    st.header("📏 OLS: A Reta de Melhor Ajuste")
    
    st.markdown("""
    **OLS (Ordinary Least Squares)** encontra a reta que **minimiza a soma dos resíduos ao quadrado**.
    
    Por que ao quadrado? Para penalizar erros grandes e evitar que erros positivos cancelem negativos.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Visualização", "🎮 Experimento Manual", "📈 Comparação"])
    
    # Dados comuns
    np.random.seed(42)
    df = make_regression_data(n=50, alpha=10, beta=2, sigma=8)
    ols = fit_ols_closed_form(df['x'].values, df['y'].values)
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            mostrar_reta = st.checkbox("Mostrar reta ajustada", value=True)
            mostrar_residuos = st.checkbox("Mostrar resíduos", value=False)
            
            st.markdown("""
            **O que são resíduos?**
            
            Resíduos são as **distâncias verticais** entre cada ponto e a reta.
            OLS encontra a reta que minimiza a soma dessas distâncias ao quadrado.
            """)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("SSE (Soma Quad. Resíduos)", f"{ols['SSE']:.1f}")
            col_m2.metric("R²", f"{ols['r_squared']:.3f}")
        
        with col2:
            fig = go.Figure()
            
            # Pontos
            fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers',
                                    marker=dict(size=10, color='#636EFA', opacity=0.7),
                                    name='Dados'))
            
            if mostrar_reta:
                x_line = np.array([df['x'].min(), df['x'].max()])
                y_line = ols['alpha'] + ols['beta'] * x_line
                fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                        line=dict(color='red', width=2),
                                        name=f"OLS: y = {ols['alpha']:.1f} + {ols['beta']:.2f}x"))
            
            if mostrar_residuos:
                for i in range(len(df)):
                    fig.add_trace(go.Scatter(
                        x=[df['x'].iloc[i], df['x'].iloc[i]],
                        y=[df['y'].iloc[i], ols['y_hat'][i]],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dot'),
                        showlegend=False
                    ))
            
            fig.update_layout(
                title="Regressão OLS",
                xaxis_title="X",
                yaxis_title="Y",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🎮 Tente Ajustar sua Própria Reta!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Sua tentativa:**")
            alpha_manual = st.slider("Seu α (intercepto)", -20.0, 50.0, 5.0, 0.5, key="alpha_manual")
            beta_manual = st.slider("Seu β (inclinação)", -1.0, 5.0, 1.5, 0.1, key="beta_manual")
            
            # Calcular SSE manual
            y_hat_manual = alpha_manual + beta_manual * df['x'].values
            sse_manual = np.sum((df['y'].values - y_hat_manual)**2)
            
            st.markdown("---")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Seu SSE", f"{sse_manual:.1f}")
            col_m2.metric("SSE do OLS", f"{ols['SSE']:.1f}")
            
            diff = sse_manual - ols['SSE']
            if diff < 1:
                st.success("🎯 Excelente! Você está muito próximo do OLS!")
            elif diff < 100:
                st.info("👍 Bom! Mas o OLS ainda é melhor.")
            else:
                st.warning(f"📈 Seu SSE está {diff:.0f} acima do OLS. Continue ajustando!")
        
        with col2:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers',
                                    marker=dict(size=10, color='#636EFA', opacity=0.7),
                                    name='Dados'))
            
            x_line = np.array([df['x'].min(), df['x'].max()])
            
            # Reta manual
            y_manual = alpha_manual + beta_manual * x_line
            fig.add_trace(go.Scatter(x=x_line, y=y_manual, mode='lines',
                                    line=dict(color='orange', width=2),
                                    name=f"Sua reta: y = {alpha_manual:.1f} + {beta_manual:.2f}x"))
            
            # Reta OLS
            y_ols = ols['alpha'] + ols['beta'] * x_line
            fig.add_trace(go.Scatter(x=x_line, y=y_ols, mode='lines',
                                    line=dict(color='red', width=2, dash='dash'),
                                    name=f"OLS: y = {ols['alpha']:.1f} + {ols['beta']:.2f}x"))
            
            fig.update_layout(title="Sua Reta vs OLS", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Por que OLS é o Melhor?")
        
        # Simular várias retas e mostrar SSE
        alphas = np.linspace(ols['alpha'] - 15, ols['alpha'] + 15, 50)
        betas = np.linspace(ols['beta'] - 1, ols['beta'] + 1, 50)
        
        sse_surface = np.zeros((len(alphas), len(betas)))
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                y_hat = a + b * df['x'].values
                sse_surface[i, j] = np.sum((df['y'].values - y_hat)**2)
        
        fig = go.Figure(data=go.Contour(
            x=betas, y=alphas, z=sse_surface,
            colorscale='Viridis',
            contours=dict(showlabels=True)
        ))
        
        # Marcar o ponto ótimo
        fig.add_trace(go.Scatter(
            x=[ols['beta']], y=[ols['alpha']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='OLS (mínimo)'
        ))
        
        fig.update_layout(
            title="Superfície de SSE: OLS está no Vale Mínimo",
            xaxis_title="β",
            yaxis_title="α",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 O ponto vermelho marca onde o SSE é mínimo — exatamente os coeficientes OLS!")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Confia que OLS dá a "melhor" reta no sentido de menor erro total
    - Entende que R² mede quanto da variação de Y é explicada por X
    """)


def render_section_S4():
    """S4: Interpretação de Resultados para Tomada de Decisão"""
    st.header("💼 Interpretação para Decisão")
    
    st.markdown("""
    O valor de uma regressão está na **interpretação dos coeficientes** para ação gerencial.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Interpretando β (efeito marginal)")
        
        st.markdown("""
        **β** responde: *"Se X aumentar em 1 unidade, quanto Y muda em média?"*
        
        Exemplos:
        - β = 5 em Marketing→Vendas: +R$1 mil em marketing → +5 vendas
        - β = -8 em Preço→Demanda: +R$1 no preço → -8 unidades vendidas
        - β = 1.2 em CAPM: +1% no mercado → +1.2% no fundo (beta > 1 = mais volátil)
        """)
        
        st.subheader("Interpretando α (intercepto)")
        
        st.warning("""
        ⚠️ **Cuidado com α!** Ele representa Y quando X=0, mas isso pode não fazer sentido.
        
        - Marketing = 0: Faz sentido (vendas sem propaganda)
        - Preço = 0: Não faz sentido (produto grátis?)
        - Idade = 0: Não faz sentido em muitos contextos
        """)
    
    with col2:
        st.subheader("📊 Exemplo: CAPM")
        
        st.markdown("""
        O **CAPM** (Capital Asset Pricing Model) usa regressão para medir risco:
        
        $$R_i - R_f = \\alpha + \\beta (R_m - R_f) + \\epsilon$$
        
        Onde:
        - β = risco sistemático (sensibilidade ao mercado)
        - α = retorno anormal (alfa de Jensen)
        """)
        
        beta_capm = st.slider("Beta do fundo", 0.5, 2.0, 1.2, 0.1, key="beta_capm")
        alpha_capm = st.slider("Alfa (% ao mês)", -1.0, 1.0, 0.2, 0.1, key="alpha_capm")
        
        df_capm = simulate_capm_data(n=60, beta_true=beta_capm, alpha_true=alpha_capm, sigma=2.0)
        ols_capm = fit_ols_closed_form(df_capm['Premio_Mercado'].values, 
                                       df_capm['Excesso_Fundo'].values)
        
        fig = px.scatter(df_capm, x='Premio_Mercado', y='Excesso_Fundo',
                        labels={'Premio_Mercado': 'Prêmio de Risco do Mercado (%)',
                               'Excesso_Fundo': 'Excesso de Retorno do Fundo (%)'},
                        title="CAPM: Fundo vs Mercado")
        
        x_line = np.array([df_capm['Premio_Mercado'].min(), df_capm['Premio_Mercado'].max()])
        y_line = ols_capm['alpha'] + ols_capm['beta'] * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                line=dict(color='red'),
                                name=f"β={ols_capm['beta']:.2f}, α={ols_capm['alpha']:.2f}"))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Beta Estimado", f"{ols_capm['beta']:.2f}", 
                     help="β > 1: mais volátil que o mercado")
        col_m2.metric("Alfa Estimado", f"{ols_capm['alpha']:.2f}%",
                     help="α > 0: retorno acima do esperado pelo risco")
    
    with st.expander("💡 Card: Implicação Gerencial"):
        st.markdown("""
        ### Como traduzir coeficientes em ação?
        
        | Contexto | Coeficiente | Ação Gerencial |
        |----------|-------------|----------------|
        | Marketing→Vendas | β = 5 | ROI: cada R$1 gera 5 vendas. Vale investir se margem > custo |
        | Preço→Demanda | β = -8 | Elasticidade: subir preço reduz volume. Otimizar ponto |
        | CAPM | β = 1.3 | Fundo amplifica mercado. Bom em alta, ruim em queda |
        | Jensen | α = 0.5% | Gestor gera valor? 0.5%/mês = 6%/ano acima do benchmark |
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Traduz β em impacto financeiro: "Se investirmos X, o retorno esperado é Y"
    - Questiona se α=0 faz sentido no contexto antes de interpretá-lo
    """)


def render_section_S5():
    """S5: Propriedades e Suposições Críticas (o 'pulo do gato')"""
    st.header("🎯 Suposições Críticas: Quando Confiar no OLS?")
    
    st.markdown("""
    OLS é **BLUE** (Best Linear Unbiased Estimator) sob certas condições.
    Em linguagem gerencial: *"a melhor estimativa linear, que acerta na média"*.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("O que significa BLUE?")
        
        st.markdown("""
        - **Unbiased (Não-viesado):** Na média, OLS acerta o valor verdadeiro de β
        - **Efficient (Eficiente):** Entre estimadores não-viesados, OLS tem menor variância
        
        **Mas isso só vale se as suposições forem verdadeiras!**
        """)
        
        st.subheader("Suposições Críticas para Decisão")
        
        st.markdown("""
        1. **Exogeneidade:** E(u|x) = 0 — o erro não é correlacionado com x
        2. **Independência:** Os erros não são correlacionados entre si
        3. **Homocedasticidade:** Variância do erro é constante
        
        A mais importante para decisão é a **exogeneidade**. Se violada, β é viesado!
        """)
    
    with col2:
        st.subheader("🔬 Simulação: Violando Exogeneidade")
        
        st.markdown("""
        O que acontece quando há uma **variável omitida** que afeta tanto X quanto Y?
        """)
        
        corr_ux = st.slider("Correlação entre erro (u) e x", -0.9, 0.9, 0.0, 0.1,
                           help="0 = exógeno (correto); ≠0 = endógeno (viés)")
        
        beta_true = 2.0
        df_endo = make_endogenous_data(n=200, beta_true=beta_true, corr_ux=corr_ux, sigma=5.0)
        ols_endo = fit_ols_closed_form(df_endo['x'].values, df_endo['y'].values)
        
        vies = ols_endo['beta'] - beta_true
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("β Verdadeiro", f"{beta_true:.2f}")
        col_m2.metric("β Estimado (OLS)", f"{ols_endo['beta']:.2f}")
        col_m3.metric("Viés", f"{vies:+.2f}", 
                     delta_color="inverse" if abs(vies) > 0.1 else "off")
        
        if abs(corr_ux) > 0.3:
            st.error(f"🚨 Viés significativo! OLS superestima/subestima o efeito real.")
        elif abs(corr_ux) > 0.1:
            st.warning("⚠️ Viés moderado. Resultados devem ser interpretados com cautela.")
        else:
            st.success("✅ Exogeneidade aproximadamente válida. OLS é confiável.")
        
        fig = px.scatter(df_endo, x='x', y='y', opacity=0.5,
                        title=f"Exogeneidade: corr(u,x) = {corr_ux}")
        
        x_line = np.array([df_endo['x'].min(), df_endo['x'].max()])
        # Reta verdadeira
        fig.add_trace(go.Scatter(x=x_line, y=10 + beta_true * x_line, mode='lines',
                                line=dict(color='green', dash='dash'),
                                name=f'Verdadeiro: β={beta_true}'))
        # Reta OLS
        fig.add_trace(go.Scatter(x=x_line, y=ols_endo['alpha'] + ols_endo['beta'] * x_line,
                                mode='lines', line=dict(color='red'),
                                name=f'OLS: β={ols_endo["beta"]:.2f}'))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📖 Nota Técnica: Por que endogeneidade causa viés?"):
        st.markdown("""
        Quando uma variável omitida (z) afeta tanto x quanto y:
        
        1. O erro u "absorve" o efeito de z
        2. Como z também afeta x, temos correlação entre u e x
        3. OLS "confunde" o efeito de z com o efeito de x
        4. Resultado: β estimado ≠ β verdadeiro
        
        **Exemplo:** Efeito de educação sobre salário. Se "habilidade" afeta ambos 
        (pessoas habilidosas estudam mais E ganham mais), OLS superestima o efeito da educação.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Pergunta: "Há variáveis omitidas que afetam tanto X quanto Y?"
    - Se sim, busca dados adicionais ou métodos alternativos (ex.: experimentos, variáveis instrumentais)
    """)


def render_section_S6():
    """S6: Inferência Estatística: Podemos Confiar no Resultado?"""
    st.header("📊 Inferência: Podemos Confiar?")
    
    st.markdown("""
    Mesmo que OLS seja não-viesado, a **estimativa tem incerteza**.
    Inferência estatística quantifica essa incerteza para decisões mais seguras.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    # Gerar dados para exemplo
    np.random.seed(42)
    df = make_regression_data(n=50, alpha=10, beta=2, sigma=10)
    ols = fit_ols_closed_form(df['x'].values, df['y'].values)
    
    with col1:
        st.subheader("Erro Padrão: Precisão da Estimativa")
        
        st.markdown("""
        O **erro padrão (SE)** mede a incerteza do coeficiente estimado.
        
        - SE menor → estimativa mais precisa
        - SE maior → mais incerteza
        
        SE depende de:
        - Variância dos erros (mais ruído → mais incerteza)
        - Tamanho da amostra (mais dados → mais precisão)
        - Variação em X (mais spread em X → mais precisão)
        """)
        
        st.metric("Erro Padrão de β", f"{ols['se_beta']:.3f}")
        
        st.subheader("Teste t: β é Significativo?")
        
        st.markdown("""
        **Hipóteses:**
        - H₀: β = 0 (X não afeta Y)
        - H₁: β ≠ 0 (X afeta Y)
        
        **Estatística t** = β̂ / SE(β̂)
        """)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Estatística t", f"{ols['t_beta']:.2f}")
        col_m2.metric("p-valor", f"{ols['p_beta']:.4f}")
        
        if ols['p_beta'] < 0.01:
            st.success("✅ Altamente significativo (p < 0.01)")
        elif ols['p_beta'] < 0.05:
            st.success("✅ Significativo (p < 0.05)")
        elif ols['p_beta'] < 0.10:
            st.warning("⚠️ Marginalmente significativo (p < 0.10)")
        else:
            st.error("❌ Não significativo (p ≥ 0.10)")
    
    with col2:
        st.subheader("Intervalo de Confiança: Margem de Segurança")
        
        st.markdown("""
        O **IC 95%** indica a faixa onde β provavelmente está.
        
        Se o IC não contém zero, β é significativo a 5%.
        """)
        
        st.metric("IC 95% para β", f"[{ols['ci_beta'][0]:.2f}, {ols['ci_beta'][1]:.2f}]")
        
        # Visualização do IC
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[ols['ci_beta'][0], ols['ci_beta'][1]],
            y=[1, 1],
            mode='lines',
            line=dict(color='blue', width=8),
            name='IC 95%'
        ))
        
        fig.add_trace(go.Scatter(
            x=[ols['beta']],
            y=[1],
            mode='markers',
            marker=dict(size=15, color='red'),
            name=f"β = {ols['beta']:.2f}"
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Intervalo de Confiança de β",
            xaxis_title="Valor de β",
            yaxis=dict(visible=False),
            height=200,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Erros de Decisão")
        
        st.markdown("""
        | Erro | Descrição | Risco de Negócio |
        |------|-----------|------------------|
        | **Tipo I** | Rejeitar H₀ quando é verdadeira | Investir em X que não funciona |
        | **Tipo II** | Não rejeitar H₀ quando é falsa | Ignorar X que funciona |
        """)
    
    st.markdown("---")
    
    st.subheader("🧪 Quiz: Decisão sob Incerteza")
    
    st.markdown("""
    **Cenário:** Você está avaliando se uma campanha de marketing (X) afeta vendas (Y).
    O p-valor do coeficiente é **0.08** e o custo de implementar a campanha é alto.
    """)
    
    resposta = st.radio(
        "O que você faz?",
        ["Implementar a campanha (β é significativo)",
         "Não implementar (p > 0.05, não é significativo)",
         "Coletar mais dados antes de decidir",
         "Depende do custo do erro Tipo I vs Tipo II"],
        key="quiz_s6"
    )
    
    if st.button("Ver feedback", key="feedback_s6"):
        if resposta == "Depende do custo do erro Tipo I vs Tipo II":
            st.success("""
            ✅ **Correto!** A decisão depende do contexto:
            
            - Se o custo de implementar sem efeito (Tipo I) é muito alto → seja conservador
            - Se o custo de perder uma oportunidade real (Tipo II) é alto → aceite p=0.08
            
            Não existe resposta universal. O limiar de 5% é convenção, não lei.
            """)
        elif resposta == "Coletar mais dados antes de decidir":
            st.info("""
            👍 **Parcialmente correto!** Mais dados reduzem incerteza, mas:
            
            - Tem custo (tempo, dinheiro)
            - Às vezes não é viável
            
            A melhor resposta considera o trade-off de erros Tipo I e II.
            """)
        else:
            st.warning("""
            ⚠️ **Incompleto.** O limiar de 5% é arbitrário. A decisão ótima depende de:
            
            - Custo de implementar algo que não funciona (Tipo I)
            - Custo de não implementar algo que funciona (Tipo II)
            
            Com p=0.08, a evidência é "marginalmente significativa" — o contexto importa!
            """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Não decide apenas pelo p-valor; considera o custo dos erros
    - Usa IC para comunicar incerteza: "O efeito está entre X e Y com 95% de confiança"
    """)


def render_section_S7():
    """S7: Casos Reais e Aplicações em Finanças"""
    st.header("💰 Aplicações em Finanças")
    
    tab1, tab2 = st.tabs(["📊 Alfa de Jensen", "🔍 Discussão: Anomalias"])
    
    with tab1:
        st.subheader("Alfa de Jensen: O Gestor Gera Valor?")
        
        st.markdown("""
        O **Alfa de Jensen** mede o retorno excedente de um fundo ajustado pelo risco:
        
        $$R_i - R_f = \\alpha + \\beta (R_m - R_f) + \\epsilon$$
        
        - **α > 0:** Gestor gera retorno acima do esperado pelo risco (skill?)
        - **α = 0:** Retorno compatível com o risco assumido
        - **α < 0:** Gestor destrói valor (ou cobra taxas altas)
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Configure o fundo:**")
            alpha_jensen = st.slider("Alfa verdadeiro (% ao mês)", -1.0, 2.0, 0.5, 0.1, key="alpha_jensen")
            beta_jensen = st.slider("Beta verdadeiro", 0.5, 1.5, 1.0, 0.1, key="beta_jensen")
            sigma_jensen = st.slider("Volatilidade idiossincrática (%)", 0.5, 4.0, 1.5, 0.25, key="sigma_jensen")
            n_meses = st.slider("Meses de histórico", 24, 120, 60, 12, key="n_meses")
        
        df_jensen = simulate_jensen_alpha(n=n_meses, alpha_true=alpha_jensen, 
                                          beta_true=beta_jensen, sigma=sigma_jensen)
        ols_jensen = fit_ols_closed_form(df_jensen['Excesso_Mercado'].values,
                                         df_jensen['Excesso_Fundo'].values)
        
        with col2:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("α Estimado", f"{ols_jensen['alpha']:.2f}%",
                         delta=f"{ols_jensen['alpha'] - alpha_jensen:+.2f}% vs real")
            col_m2.metric("p-valor (α)", f"{ols_jensen['p_alpha']:.3f}")
            col_m3.metric("β Estimado", f"{ols_jensen['beta']:.2f}")
            
            # Interpretação
            if ols_jensen['p_alpha'] < 0.05 and ols_jensen['alpha'] > 0:
                st.success("✅ Alfa positivo e significativo: evidência de skill!")
            elif ols_jensen['alpha'] > 0 and ols_jensen['p_alpha'] >= 0.05:
                st.warning("⚠️ Alfa positivo mas não significativo: pode ser sorte")
            elif ols_jensen['alpha'] <= 0:
                st.error("❌ Alfa zero ou negativo: sem evidência de geração de valor")
        
        fig = px.scatter(df_jensen, x='Excesso_Mercado', y='Excesso_Fundo',
                        labels={'Excesso_Mercado': 'Excesso de Retorno do Mercado (%)',
                               'Excesso_Fundo': 'Excesso de Retorno do Fundo (%)'},
                        title="Análise de Alfa de Jensen")
        
        x_line = np.array([df_jensen['Excesso_Mercado'].min(), df_jensen['Excesso_Mercado'].max()])
        fig.add_trace(go.Scatter(x=x_line, y=ols_jensen['alpha'] + ols_jensen['beta'] * x_line,
                                mode='lines', line=dict(color='red'),
                                name=f"α={ols_jensen['alpha']:.2f}%, β={ols_jensen['beta']:.2f}"))
        # Linha de mercado (alfa = 0)
        fig.add_trace(go.Scatter(x=x_line, y=ols_jensen['beta'] * x_line,
                                mode='lines', line=dict(color='gray', dash='dash'),
                                name="α=0 (benchmark)"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("💼 Implicação: O que muda na avaliação do gestor?"):
            st.markdown(f"""
            **Resultados da análise:**
            - Alfa estimado: {ols_jensen['alpha']:.2f}% ao mês ({ols_jensen['alpha']*12:.1f}% ao ano)
            - IC 95%: [{ols_jensen['ci_alpha'][0]:.2f}%, {ols_jensen['ci_alpha'][1]:.2f}%]
            
            **Decisão de alocação:**
            - Se α > 0 e significativo: considerar aumentar alocação
            - Se α ≈ 0: avaliar se a taxa de administração justifica
            - Se α < 0: questionar a permanência no fundo
            
            **Cuidados:**
            - Alfa passado não garante alfa futuro
            - Períodos curtos têm alta incerteza
            - Considerar custos de transação e taxas
            """)
    
    with tab2:
        st.subheader("🔍 Provocação: Anomalias de Mercado")
        
        st.markdown("""
        Se mercados são eficientes, **não deveria haver alfa consistente**. 
        Mas a literatura documenta várias "anomalias":
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Anomalias Clássicas:**
            
            - **Efeito Momentum:** Ativos que subiram continuam subindo no curto prazo
            - **Efeito Valor:** Ações "baratas" (P/L baixo) superam "caras"
            - **Efeito Tamanho:** Small caps historicamente superam large caps
            - **Sobre-reação:** Mercado exagera em notícias, depois corrige
            """)
        
        with col2:
            st.markdown("""
            **Interpretações:**
            
            1. **Risco:** Anomalias são prêmios por riscos não capturados pelo CAPM
            2. **Comportamento:** Vieses cognitivos dos investidores
            3. **Data mining:** Padrões espúrios encontrados no passado
            4. **Limites à arbitragem:** Custos impedem exploração
            """)
        
        st.info("""
        💡 **Conexão com regressão:** Anomalias são detectadas via regressão — se α ≠ 0 
        sistematicamente para certas estratégias, há "retorno anormal". A questão é: 
        é skill, risco ou ilusão estatística?
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa alfa de Jensen para avaliar gestores ativos vs passivos
    - Questiona: "O alfa é estatisticamente significativo? Persiste no tempo?"
    """)


def render_section_S8():
    """S8: Resumo Executivo e Ponte para o Próximo Módulo"""
    st.header("📋 Resumo Executivo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### O que Aprendemos sobre Regressão Linear
        
        ✅ **O que regressão faz:**
        - Quantifica a relação entre variáveis (efeito de X sobre Y)
        - Fornece estimativas pontuais (β) e medidas de incerteza (SE, IC)
        - Permite previsão: dado X, qual o Y esperado?
        
        ❌ **O que regressão NÃO faz:**
        - Não prova causalidade automaticamente (correlação ≠ causação)
        - Não funciona bem com suposições violadas (endogeneidade → viés)
        - Não substitui entendimento do negócio
        
        📖 **Como interpretar:**
        - β = efeito marginal: +1 em X → +β em Y (em média)
        - α = intercepto (cuidado com interpretação quando X=0 não faz sentido)
        - R² = % da variação de Y explicada por X
        
        🎯 **Quando confiar:**
        - Exogeneidade válida (erro não correlacionado com X)
        - Amostra representativa e suficientemente grande
        - p-valor e IC indicam significância estatística
        
        ⚠️ **Riscos comuns:**
        - Variáveis omitidas que causam viés
        - Confundir significância estatística com relevância prática
        - Extrapolar além do range dos dados
        """)
    
    with col2:
        st.markdown("### Métricas-Chave")
        
        # Exemplo com dados
        df = make_regression_data(n=100, alpha=10, beta=2, sigma=8)
        ols = fit_ols_closed_form(df['x'].values, df['y'].values)
        
        st.metric("α (intercepto)", f"{ols['alpha']:.2f}")
        st.metric("β (efeito)", f"{ols['beta']:.2f}")
        st.metric("R²", f"{ols['r_squared']:.1%}")
        st.metric("SE(β)", f"{ols['se_beta']:.3f}")
        st.metric("p-valor(β)", f"{ols['p_beta']:.4f}")
    
    st.markdown("---")
    
    st.subheader("🔜 Próximo Módulo: Extensões e Diagnósticos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Regressão Múltipla:**
        - Múltiplos X's no mesmo modelo
        - Controle de variáveis confundidoras
        - Interpretação ceteris paribus
        """)
    
    with col2:
        st.markdown("""
        **Diagnósticos:**
        - Testes de heterocedasticidade
        - Detecção de multicolinearidade
        - Análise de resíduos
        """)
    
    with col3:
        st.markdown("""
        **Extensões:**
        - Variáveis dummy (categóricas)
        - Transformações (log, quadrático)
        - Interações entre variáveis
        """)
    
    st.success("""
    🎓 **Mensagem final:** Regressão é ferramenta poderosa, mas requer julgamento crítico.
    Entenda as suposições, questione a exogeneidade, e sempre conecte os números à decisão de negócio.
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa regressão como ponto de partida para análise quantitativa
    - Exige robustez: "Os resultados mudam se incluirmos outras variáveis?"
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("📈 Módulo 2: Regressão Linear Clássica (CLRM)")
    st.markdown("**Laboratório de Econometria** | Modelando Relações para Decisão")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Apresentar regressão como ferramenta para modelar relações e apoiar decisões
        - Ensinar a **leitura gerencial** de uma regressão: coeficientes, resíduos, R²
        - Introduzir OLS como "reta de melhor ajuste" via minimização de resíduos
        - Explicar **inferência estatística**: erro padrão, teste t, p-valor, intervalos de confiança
        - Conectar a aplicações em finanças: CAPM, Alfa de Jensen
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "📈 Introdução e Motivação",
        "S2": "📐 Modelo de Regressão Simples",
        "S3": "📏 OLS: Reta de Melhor Ajuste",
        "S4": "💼 Interpretação para Decisão",
        "S5": "🎯 Suposições Críticas",
        "S6": "📊 Inferência Estatística",
        "S7": "💰 Aplicações em Finanças",
        "S8": "📋 Resumo e Próximos Passos"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Dica:** Use os controles interativos 
    para experimentar com diferentes parâmetros.
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
    # Configuração da página (apenas quando executado diretamente)
    try:
        st.set_page_config(
            page_title="Módulo 2: Regressão Linear (CLRM)",
            page_icon="📈",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()