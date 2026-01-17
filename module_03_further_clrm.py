"""
Laboratório de Econometria - Module 3: Further Development of CLRM
Aplicativo educacional interativo para extensões da regressão linear.
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
# FUNÇÕES AUXILIARES PARA GERAÇÃO DE DADOS E CÁLCULOS
# =============================================================================

@st.cache_data
def make_multireg_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Gera dados para regressão múltipla: retornos vs inflação, produção, prêmio de risco."""
    np.random.seed(seed)
    
    # Variáveis explicativas
    inflacao = np.random.normal(4, 1.5, n)  # Inflação %
    producao = np.random.normal(2, 1.0, n)  # Crescimento da produção %
    premio_risco = np.random.normal(5, 2.0, n)  # Prêmio de risco %
    
    # Coeficientes verdadeiros
    alpha_true = 2.0
    beta_inflacao = -0.8  # Inflação reduz retornos
    beta_producao = 1.5   # Produção aumenta retornos
    beta_premio = 0.6     # Prêmio de risco aumenta retornos
    
    # Erro
    erro = np.random.normal(0, 2, n)
    
    # Retorno
    retorno = (alpha_true + beta_inflacao * inflacao + 
               beta_producao * producao + beta_premio * premio_risco + erro)
    
    return pd.DataFrame({
        'Retorno': retorno,
        'Inflacao': inflacao,
        'Producao': producao,
        'Premio_Risco': premio_risco
    })


def fit_ols_multiple(X: np.ndarray, y: np.ndarray) -> dict:
    """Calcula OLS múltiplo via fórmula matricial (X'X)^(-1)X'y."""
    n, k = X.shape
    
    # Adicionar constante se não existir
    if not np.allclose(X[:, 0], 1):
        X = np.column_stack([np.ones(n), X])
        k = X.shape[1]
    
    # Coeficientes: (X'X)^(-1)X'y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ X.T @ y
    
    # Valores ajustados e resíduos
    y_hat = X @ beta_hat
    residuals = y - y_hat
    
    # Soma dos quadrados
    SSE = np.sum(residuals**2)  # Unrestricted RSS
    SST = np.sum((y - np.mean(y))**2)
    SSR = SST - SSE
    
    # R² e R² ajustado
    r_squared = 1 - SSE / SST
    r_squared_adj = 1 - (SSE / (n - k)) / (SST / (n - 1))
    
    # Variância dos resíduos
    s2 = SSE / (n - k)
    
    # Erros padrão dos coeficientes
    var_beta = s2 * np.diag(XtX_inv)
    se_beta = np.sqrt(var_beta)
    
    # Estatísticas t e p-valores
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
        'SSR': SSR,
        'r_squared': r_squared,
        'r_squared_adj': r_squared_adj,
        's2': s2,
        'n': n,
        'k': k
    }


def compute_f_test(sse_unrestricted: float, sse_restricted: float, 
                   n: int, k_unrestricted: int, q: int) -> dict:
    """Calcula o teste F para restrições lineares."""
    # q = número de restrições (diferença no número de parâmetros)
    # F = [(RRSS - URSS)/q] / [URSS/(n-k)]
    
    f_stat = ((sse_restricted - sse_unrestricted) / q) / (sse_unrestricted / (n - k_unrestricted))
    p_value = 1 - stats.f.cdf(f_stat, q, n - k_unrestricted)
    
    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'df1': q,
        'df2': n - k_unrestricted
    }


@st.cache_data
def make_dummy_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Gera dados para modelo hedônico de imóveis com dummies."""
    np.random.seed(seed)
    
    # Área do imóvel
    area = np.random.uniform(40, 200, n)
    
    # Bairro (categórica)
    bairros = np.random.choice(['Centro', 'Zona Sul', 'Zona Norte'], n, p=[0.3, 0.4, 0.3])
    
    # Garagem (dummy)
    garagem = np.random.choice([0, 1], n, p=[0.4, 0.6])
    
    # Efeitos verdadeiros
    beta_area = 5000  # R$ por m²
    preco_base = 100000
    efeito_zona_sul = 150000  # Premium Zona Sul
    efeito_zona_norte = -50000  # Desconto Zona Norte
    efeito_garagem = 80000
    
    # Preço
    preco = preco_base + beta_area * area + efeito_garagem * garagem
    preco += np.where(bairros == 'Zona Sul', efeito_zona_sul, 0)
    preco += np.where(bairros == 'Zona Norte', efeito_zona_norte, 0)
    preco += np.random.normal(0, 50000, n)  # Ruído
    
    return pd.DataFrame({
        'Preco': preco,
        'Area': area,
        'Bairro': bairros,
        'Garagem': garagem
    })


def fit_quantile_regression(X: np.ndarray, y: np.ndarray, tau: float = 0.5, 
                           max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Regressão quantílica via IRLS simplificado."""
    n, k = X.shape
    
    # Inicialização com OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    for _ in range(max_iter):
        residuals = y - X @ beta
        
        # Pesos para IRLS
        weights = np.where(residuals >= 0, tau, 1 - tau)
        weights = weights / (np.abs(residuals) + 1e-6)
        
        # Weighted least squares
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        
        beta_new = np.linalg.solve(XtWX + 1e-8 * np.eye(k), XtWy)
        
        if np.max(np.abs(beta_new - beta)) < tol:
            break
        beta = beta_new
    
    return beta


@st.cache_data
def make_quantile_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Gera dados com heterocedasticidade para demonstrar regressão quantílica."""
    np.random.seed(seed)
    
    x = np.random.uniform(1, 10, n)
    
    # Erro heterocedástico: variância aumenta com x
    erro = np.random.normal(0, 1, n) * (0.5 + 0.5 * x)
    
    y = 2 + 1.5 * x + erro
    
    return pd.DataFrame({'x': x, 'y': y})


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: Regressão Linear Múltipla — Intuição de Negócios"""
    st.header("📊 Regressão Múltipla: Efeitos Parciais")
    
    st.markdown("""
    Na prática, resultados dependem de **múltiplos fatores**. 
    A regressão múltipla permite isolar o efeito de cada variável, 
    mantendo as outras constantes (*ceteris paribus*).
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Exemplo: Retornos de Ações")
        
        st.markdown("""
        O retorno de um portfólio pode depender de:
        - **Inflação:** Corrói retornos reais
        - **Produção Industrial:** Indica crescimento econômico
        - **Prêmio de Risco:** Compensação por volatilidade
        
        O modelo múltiplo:
        
        $$Retorno = \\alpha + \\beta_1 \\cdot Inflação + \\beta_2 \\cdot Produção + \\beta_3 \\cdot Prêmio + u$$
        """)
        
        st.info("""
        💡 **Ceteris Paribus:** Cada β mede o efeito de sua variável 
        *mantendo as outras constantes*. Isso isola contribuições individuais.
        """)
        
        # Controles para incluir/excluir variáveis
        st.subheader("Selecione as variáveis:")
        usar_inflacao = st.checkbox("Incluir Inflação", value=True)
        usar_producao = st.checkbox("Incluir Produção", value=True)
        usar_premio = st.checkbox("Incluir Prêmio de Risco", value=True)
    
    with col2:
        # Gerar dados
        df = make_multireg_data(n=200)
        
        # Construir matriz X com variáveis selecionadas
        variaveis = []
        nomes = ['Intercepto']
        X = np.ones((len(df), 1))
        
        if usar_inflacao:
            X = np.column_stack([X, df['Inflacao'].values])
            nomes.append('Inflação')
            variaveis.append('Inflacao')
        if usar_producao:
            X = np.column_stack([X, df['Producao'].values])
            nomes.append('Produção')
            variaveis.append('Producao')
        if usar_premio:
            X = np.column_stack([X, df['Premio_Risco'].values])
            nomes.append('Prêmio Risco')
            variaveis.append('Premio_Risco')
        
        if len(variaveis) == 0:
            st.warning("⚠️ Selecione ao menos uma variável explicativa.")
        else:
            # Ajustar modelo
            ols = fit_ols_multiple(X, df['Retorno'].values)
            
            # Mostrar resultados
            st.subheader("Resultados da Regressão")
            
            results_df = pd.DataFrame({
                'Variável': nomes,
                'Coeficiente': ols['beta'],
                'Erro Padrão': ols['se'],
                't-stat': ols['t_stats'],
                'p-valor': ols['p_values']
            })
            results_df['Coeficiente'] = results_df['Coeficiente'].round(3)
            results_df['Erro Padrão'] = results_df['Erro Padrão'].round(3)
            results_df['t-stat'] = results_df['t-stat'].round(2)
            results_df['p-valor'] = results_df['p-valor'].round(4)
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("R²", f"{ols['r_squared']:.3f}")
            col_m2.metric("R² Ajustado", f"{ols['r_squared_adj']:.3f}")
    
    # Visualização
    if len(variaveis) > 0:
        st.subheader("Visualização: Efeito Parcial")
        
        var_plot = st.selectbox("Variável para visualizar:", variaveis)
        
        fig = px.scatter(df, x=var_plot, y='Retorno', opacity=0.6,
                        title=f"Retorno vs {var_plot}")
        
        # Adicionar linha de regressão parcial (simplificada)
        x_var = df[var_plot].values
        slope_idx = nomes.index(var_plot.replace('_', ' ').replace('Inflacao', 'Inflação').replace('Producao', 'Produção').replace('Premio Risco', 'Prêmio Risco'))
        
        x_line = np.array([x_var.min(), x_var.max()])
        y_line = ols['beta'][0] + ols['beta'][slope_idx] * x_line
        
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                line=dict(color='red', width=2),
                                name=f"β = {ols['beta'][slope_idx]:.2f}"))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📖 Nota Técnica: Forma Matricial"):
        st.markdown("""
        Computacionalmente, a regressão múltipla é expressa como:
        
        $$\\mathbf{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\mathbf{u}$$
        
        Onde:
        - **y** é o vetor n×1 de observações da variável dependente
        - **X** é a matriz n×k de variáveis explicativas (incluindo constante)
        - **β** é o vetor k×1 de coeficientes
        - **u** é o vetor n×1 de erros
        
        A solução OLS é: $\\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}'\\mathbf{X})^{-1}\\mathbf{X}'\\mathbf{y}$
        
        Esta fórmula é a base de todos os softwares estatísticos.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Isola o efeito de cada driver controlando pelos demais
    - Responde: "Se a inflação subir 1%, qual o impacto no retorno, mantendo produção e prêmio constantes?"
    """)


def render_section_S2():
    """S2: Testes de Hipóteses Múltiplas — Teste F"""
    st.header("🧪 Teste F: Testando Variáveis em Conjunto")
    
    st.markdown("""
    Às vezes queremos testar se **um grupo de variáveis** é conjuntamente significativo,
    não apenas individualmente. O **Teste F** compara dois modelos:
    
    - **Modelo Irrestrito:** Inclui todas as variáveis
    - **Modelo Restrito:** Exclui as variáveis testadas (impõe β = 0)
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Lógica do Teste F")
        
        st.markdown("""
        **Pergunta:** As variáveis excluídas melhoram significativamente o ajuste?
        
        **Estatística F:**
        
        $$F = \\frac{(SSE_R - SSE_{UR})/q}{SSE_{UR}/(n-k)}$$
        
        Onde:
        - SSE_R = Soma dos quadrados dos resíduos (restrito)
        - SSE_UR = Soma dos quadrados dos resíduos (irrestrito)
        - q = número de restrições (variáveis excluídas)
        - k = número de parâmetros no modelo irrestrito
        """)
        
        st.info("""
        💡 **Intuição:** Se excluir variáveis aumenta muito o erro (SSE), 
        elas são conjuntamente importantes. F alto → rejeita H₀.
        """)
    
    with col2:
        st.subheader("Simulação Interativa")
        
        # Gerar dados
        df = make_multireg_data(n=200)
        
        st.markdown("**H₀:** Produção e Prêmio de Risco não afetam retornos")
        st.markdown("**H₁:** Pelo menos um deles afeta")
        
        # Modelo irrestrito (todas as variáveis)
        X_ur = np.column_stack([
            np.ones(len(df)),
            df['Inflacao'].values,
            df['Producao'].values,
            df['Premio_Risco'].values
        ])
        ols_ur = fit_ols_multiple(X_ur, df['Retorno'].values)
        
        # Modelo restrito (só inflação)
        X_r = np.column_stack([
            np.ones(len(df)),
            df['Inflacao'].values
        ])
        ols_r = fit_ols_multiple(X_r, df['Retorno'].values)
        
        # Teste F
        f_test = compute_f_test(
            sse_unrestricted=ols_ur['SSE'],
            sse_restricted=ols_r['SSE'],
            n=ols_ur['n'],
            k_unrestricted=ols_ur['k'],
            q=2  # Testando 2 variáveis
        )
        
        # Visualização
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("SSE Restrito", f"{ols_r['SSE']:.1f}")
        col_m2.metric("SSE Irrestrito", f"{ols_ur['SSE']:.1f}")
        
        col_m3, col_m4 = st.columns(2)
        col_m3.metric("Estatística F", f"{f_test['f_stat']:.2f}")
        col_m4.metric("p-valor", f"{f_test['p_value']:.4f}")
        
        if f_test['p_value'] < 0.05:
            st.success("✅ Rejeita H₀: Produção e/ou Prêmio são significativos!")
        else:
            st.warning("⚠️ Não rejeita H₀: Evidência insuficiente")
    
    # Gráfico comparativo
    st.subheader("Comparação Visual: URSS vs RRSS")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Modelo Restrito\n(só Inflação)', 'Modelo Irrestrito\n(todas variáveis)'],
        y=[ols_r['SSE'], ols_ur['SSE']],
        marker_color=['#EF553B', '#636EFA'],
        text=[f"{ols_r['SSE']:.0f}", f"{ols_ur['SSE']:.0f}"],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Soma dos Quadrados dos Resíduos (SSE)",
        yaxis_title="SSE",
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📖 Nota Técnica: Relação entre Teste t e Teste F"):
        st.markdown("""
        **Para uma única restrição (q=1):**
        
        $$F = t^2$$
        
        O teste F com uma restrição é equivalente ao teste t bilateral.
        
        **Quando usar cada um:**
        - **Teste t:** Para testar uma variável individualmente
        - **Teste F:** Para testar múltiplas variáveis conjuntamente
        
        **Exemplo prático:**
        - Teste t: "A inflação afeta retornos?"
        - Teste F: "Inflação, produção e prêmio afetam retornos conjuntamente?"
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Testa se um conjunto de variáveis é relevante antes de incluí-las no modelo
    - Compara modelos alternativos de forma rigorosa
    """)


def render_section_S3():
    """S3: Qualidade do Ajuste e Seleção de Modelos"""
    st.header("📈 R² vs R² Ajustado: Evitando Overfitting")
    
    st.markdown("""
    Adicionar variáveis **sempre** aumenta R², mesmo que sejam irrelevantes.
    O **R² ajustado** penaliza a inclusão de variáveis desnecessárias.
    """)
    
    tab1, tab2 = st.tabs(["📊 Comparação", "⚠️ Simulação de Overfitting"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("R² vs R² Ajustado")
            
            st.markdown("""
            **R² (Coeficiente de Determinação):**
            
            $$R^2 = 1 - \\frac{SSE}{SST}$$
            
            Mede a proporção da variância explicada, mas **sempre aumenta** com mais variáveis.
            
            **R² Ajustado:**
            
            $$\\bar{R}^2 = 1 - \\frac{SSE/(n-k)}{SST/(n-1)}$$
            
            Penaliza a inclusão de variáveis que não melhoram o ajuste proporcionalmente.
            """)
            
            st.info("""
            💡 **Regra prática:** Se R² aumenta mas R² ajustado cai, 
            a nova variável provavelmente não é útil.
            """)
        
        with col2:
            st.subheader("Inclusão Incremental de Variáveis")
            
            df = make_multireg_data(n=150)
            
            # Modelos com diferentes números de variáveis
            modelos = []
            
            # Modelo 1: só constante
            X1 = np.ones((len(df), 1))
            ols1 = fit_ols_multiple(X1, df['Retorno'].values)
            modelos.append(('Só Constante', 1, 0, ols1['r_squared_adj']))
            
            # Modelo 2: + Inflação
            X2 = np.column_stack([np.ones(len(df)), df['Inflacao'].values])
            ols2 = fit_ols_multiple(X2, df['Retorno'].values)
            modelos.append(('+ Inflação', 2, ols2['r_squared'], ols2['r_squared_adj']))
            
            # Modelo 3: + Produção
            X3 = np.column_stack([X2, df['Producao'].values])
            ols3 = fit_ols_multiple(X3, df['Retorno'].values)
            modelos.append(('+ Produção', 3, ols3['r_squared'], ols3['r_squared_adj']))
            
            # Modelo 4: + Prêmio
            X4 = np.column_stack([X3, df['Premio_Risco'].values])
            ols4 = fit_ols_multiple(X4, df['Retorno'].values)
            modelos.append(('+ Prêmio', 4, ols4['r_squared'], ols4['r_squared_adj']))
            
            df_modelos = pd.DataFrame(modelos, 
                                      columns=['Modelo', 'k', 'R²', 'R² Ajustado'])
            st.dataframe(df_modelos.round(4), use_container_width=True, hide_index=True)
            
            # Gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_modelos['k'], y=df_modelos['R²'],
                                    mode='lines+markers', name='R²',
                                    line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df_modelos['k'], y=df_modelos['R² Ajustado'],
                                    mode='lines+markers', name='R² Ajustado',
                                    line=dict(color='red', dash='dash')))
            fig.update_layout(
                title="R² vs R² Ajustado por Número de Variáveis",
                xaxis_title="Número de Parâmetros (k)",
                yaxis_title="Valor",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("⚠️ Demonstração: Overfitting com Ruído")
        
        st.markdown("""
        O que acontece quando adicionamos variáveis **completamente aleatórias**?
        """)
        
        n_ruido = st.slider("Número de variáveis aleatórias (ruído)", 0, 20, 5)
        
        df = make_multireg_data(n=100)
        y = df['Retorno'].values
        
        # Modelo base (só variáveis reais)
        X_base = np.column_stack([
            np.ones(len(df)),
            df['Inflacao'].values,
            df['Producao'].values,
            df['Premio_Risco'].values
        ])
        
        # Adicionar variáveis de ruído
        np.random.seed(123)
        X_ruido = X_base.copy()
        for i in range(n_ruido):
            ruido = np.random.normal(0, 1, len(df))
            X_ruido = np.column_stack([X_ruido, ruido])
        
        ols_base = fit_ols_multiple(X_base, y)
        ols_ruido = fit_ols_multiple(X_ruido, y)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² (base)", f"{ols_base['r_squared']:.4f}")
        col2.metric("R² (+ ruído)", f"{ols_ruido['r_squared']:.4f}",
                   delta=f"+{(ols_ruido['r_squared'] - ols_base['r_squared']):.4f}")
        col3.metric("R² Adj (base)", f"{ols_base['r_squared_adj']:.4f}")
        col4.metric("R² Adj (+ ruído)", f"{ols_ruido['r_squared_adj']:.4f}",
                   delta=f"{(ols_ruido['r_squared_adj'] - ols_base['r_squared_adj']):.4f}")
        
        if ols_ruido['r_squared'] > ols_base['r_squared'] and ols_ruido['r_squared_adj'] < ols_base['r_squared_adj']:
            st.error("""
            🚨 **Overfitting detectado!** R² aumentou, mas R² ajustado caiu.
            As variáveis de ruído não têm valor preditivo real.
            """)
        elif n_ruido > 0:
            st.warning("⚠️ R² sempre aumenta com mais variáveis, mesmo sem valor real.")
    
    with st.expander("📖 Nota: Modelos Não-Aninhados"):
        st.markdown("""
        **Modelos aninhados:** Um é caso especial do outro (ex.: com/sem uma variável).
        - Use Teste F para comparar.
        
        **Modelos não-aninhados:** Variáveis diferentes, nenhum é caso especial.
        - Exemplo: Modelo A usa inflação; Modelo B usa câmbio.
        - Não dá para usar Teste F diretamente.
        - Alternativas: AIC, BIC, validação cruzada (fora do escopo).
        
        **Regra prática:** Prefira modelos mais simples que explicam bem os dados.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Monitora R² ajustado para evitar modelos inflados
    - Questiona: "Essa variável extra realmente melhora a previsão?"
    """)


def render_section_S4():
    """S4: Variáveis Qualitativas (Dummies) e Modelos Hedônicos"""
    st.header("🏠 Variáveis Dummy e Modelos Hedônicos")
    
    st.markdown("""
    **Variáveis dummy** (0/1) permitem incluir categorias qualitativas na regressão.
    **Modelos hedônicos** decompõem o preço de um bem em seus atributos.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Exemplo: Precificação de Imóveis")
        
        st.markdown("""
        O preço de um apartamento depende de:
        - **Área** (m²) — variável contínua
        - **Bairro** (Centro, Zona Sul, Zona Norte) — variável categórica
        - **Garagem** (sim/não) — variável binária
        
        **Como incluir bairro na regressão?**
        
        Criamos dummies:
        - D_ZonaSul = 1 se Zona Sul, 0 caso contrário
        - D_ZonaNorte = 1 se Zona Norte, 0 caso contrário
        - Centro é a **categoria de referência** (quando ambas = 0)
        """)
        
        st.warning("""
        ⚠️ **Armadilha da Variável Dummy:** Nunca inclua dummies para todas as categorias!
        Se incluir D_Centro também, há multicolinearidade perfeita (soma = 1).
        Sempre omita uma categoria como referência.
        """)
    
    with col2:
        st.subheader("Dados Simulados")
        
        df = make_dummy_data(n=300)
        
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown(f"""
        **Estatísticas:**
        - Observações: {len(df)}
        - Área média: {df['Area'].mean():.1f} m²
        - Preço médio: R$ {df['Preco'].mean():,.0f}
        """)
    
    # Ajustar modelo
    st.subheader("Modelo Hedônico Estimado")
    
    # Criar dummies
    df['D_ZonaSul'] = (df['Bairro'] == 'Zona Sul').astype(int)
    df['D_ZonaNorte'] = (df['Bairro'] == 'Zona Norte').astype(int)
    
    X = np.column_stack([
        np.ones(len(df)),
        df['Area'].values,
        df['Garagem'].values,
        df['D_ZonaSul'].values,
        df['D_ZonaNorte'].values
    ])
    
    ols = fit_ols_multiple(X, df['Preco'].values)
    
    nomes = ['Intercepto', 'Área (m²)', 'Garagem', 'Zona Sul', 'Zona Norte']
    results_df = pd.DataFrame({
        'Variável': nomes,
        'Coeficiente': ols['beta'],
        'Erro Padrão': ols['se'],
        'p-valor': ols['p_values']
    })
    results_df['Coeficiente'] = results_df['Coeficiente'].apply(lambda x: f"R$ {x:,.0f}")
    results_df['Erro Padrão'] = results_df['Erro Padrão'].apply(lambda x: f"R$ {x:,.0f}")
    results_df['p-valor'] = results_df['p-valor'].round(4)
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("R²", f"{ols['r_squared']:.3f}")
    col_m2.metric("R² Ajustado", f"{ols['r_squared_adj']:.3f}")
    
    # Interpretação
    with st.expander("💡 Como Interpretar os Coeficientes?"):
        st.markdown(f"""
        **Interpretação dos resultados:**
        
        - **Área:** Cada m² adicional aumenta o preço em ~R$ {ols['beta'][1]:,.0f}
        - **Garagem:** Ter garagem adiciona ~R$ {ols['beta'][2]:,.0f} ao preço
        - **Zona Sul:** Em média, R$ {ols['beta'][3]:,.0f} mais caro que o Centro
        - **Zona Norte:** Em média, R$ {ols['beta'][4]:,.0f} em relação ao Centro
        
        **Visualização: Dummies deslocam o intercepto**
        
        - Imóvel no Centro: Preço = {ols['beta'][0]:,.0f} + {ols['beta'][1]:,.0f}×Área + {ols['beta'][2]:,.0f}×Garagem
        - Imóvel na Zona Sul: Preço = {ols['beta'][0] + ols['beta'][3]:,.0f} + {ols['beta'][1]:,.0f}×Área + ...
        """)
    
    # Gráfico
    fig = px.scatter(df, x='Area', y='Preco', color='Bairro',
                    symbol='Garagem',
                    labels={'Area': 'Área (m²)', 'Preco': 'Preço (R$)'},
                    title="Preço vs Área por Bairro e Garagem")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Decompõe preços em atributos para precificação estratégica
    - Quantifica o "prêmio" de localização ou características
    """)


def render_section_S5():
    """S5: Indo Além da Média — Regressão Quantílica"""
    st.header("📊 Regressão Quantílica: Além da Média")
    
    st.markdown("""
    OLS estima o efeito sobre a **média** de Y. Mas e se quisermos entender o efeito
    sobre os **extremos**? A **regressão quantílica** estima efeitos em diferentes pontos da distribuição.
    """)
    
    tab1, tab2 = st.tabs(["📈 Comparação Visual", "💼 Aplicação: Risco"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("OLS vs Quantis")
            
            st.markdown("""
            **OLS:** Minimiza soma dos quadrados dos resíduos
            - Estima E(Y|X) — a média condicional
            
            **Regressão Quantílica:** Minimiza soma ponderada dos desvios absolutos
            - Estima Q_τ(Y|X) — o quantil τ condicional
            
            **Exemplos de quantis:**
            - τ = 0.10: Percentil 10 (cauda inferior)
            - τ = 0.50: Mediana
            - τ = 0.90: Percentil 90 (cauda superior)
            """)
            
            st.info("""
            💡 **Utilidade:** Se o efeito de X varia ao longo da distribuição de Y,
            OLS dá uma visão incompleta. Quantílica revela heterogeneidade.
            """)
        
        with col2:
            # Gerar dados com heterocedasticidade
            df = make_quantile_data(n=300)
            
            # Ajustar OLS
            X = np.column_stack([np.ones(len(df)), df['x'].values])
            ols = fit_ols_multiple(X, df['y'].values)
            
            # Ajustar regressões quantílicas
            quantis = [0.10, 0.50, 0.90]
            betas_quantil = {}
            for tau in quantis:
                beta_q = fit_quantile_regression(X, df['y'].values, tau=tau)
                betas_quantil[tau] = beta_q
            
            st.markdown("**Coeficientes Estimados:**")
            
            comp_df = pd.DataFrame({
                'Método': ['OLS (média)', 'Quantil 10%', 'Quantil 50%', 'Quantil 90%'],
                'Intercepto': [ols['beta'][0], betas_quantil[0.10][0], 
                              betas_quantil[0.50][0], betas_quantil[0.90][0]],
                'β (efeito de X)': [ols['beta'][1], betas_quantil[0.10][1],
                                   betas_quantil[0.50][1], betas_quantil[0.90][1]]
            })
            comp_df = comp_df.round(3)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Gráfico com múltiplas retas
        st.subheader("Visualização: Múltiplas Retas por Quantil")
        
        fig = px.scatter(df, x='x', y='y', opacity=0.5,
                        title="OLS vs Regressão Quantílica")
        
        x_line = np.array([df['x'].min(), df['x'].max()])
        
        # Reta OLS
        fig.add_trace(go.Scatter(x=x_line, y=ols['beta'][0] + ols['beta'][1] * x_line,
                                mode='lines', line=dict(color='black', width=3),
                                name='OLS (média)'))
        
        # Retas quantílicas
        colors = {0.10: 'blue', 0.50: 'green', 0.90: 'red'}
        for tau, beta_q in betas_quantil.items():
            fig.add_trace(go.Scatter(
                x=x_line, y=beta_q[0] + beta_q[1] * x_line,
                mode='lines', line=dict(color=colors[tau], dash='dash'),
                name=f'Quantil {int(tau*100)}%'
            ))
                    
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Observe:** As retas divergem porque a variância aumenta com X.
        O efeito no P90 é maior que no P10 — heterogeneidade!
        """)
    
    with tab2:
        st.subheader("💼 Aplicação: Análise de Risco")
        
        st.markdown("""
        **Cenário:** Você quer entender como um fator de risco (X) afeta 
        os retornos de um portfólio, especialmente nas caudas.
        
        - **Cauda inferior (P10):** Perdas extremas
        - **Mediana (P50):** Retorno típico
        - **Cauda superior (P90):** Ganhos extremos
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Interpretação para gestão de risco:**
            
            Se β no P10 < β na média:
            - O fator X tem **menos impacto** nas perdas extremas
            - Proteção parcial em cenários ruins
            
            Se β no P90 > β na média:
            - O fator X **amplifica ganhos** nos bons cenários
            - Potencial de upside
            
            **Ação gerencial:**
            - Use P10 para stress tests e VaR
            - Use P90 para cenários otimistas
            - Média sozinha pode esconder riscos assimétricos
            """)
        
        with col2:
            # Quiz
            st.subheader("🧪 Quiz Rápido")
            
            st.markdown("""
            Um gestor de risco estimou que:
            - β no P10 = 0.5
            - β na média = 1.0
            - β no P90 = 1.5
            
            **O que isso significa?**
            """)
            
            resposta = st.radio(
                "Selecione:",
                ["O fator X tem efeito constante na distribuição",
                 "O fator X amplifica extremos (mais upside que downside)",
                 "O fator X protege nas caudas",
                 "Não é possível interpretar"],
                key="quiz_s5"
            )
            
            if st.button("Ver resposta", key="btn_s5"):
                if resposta == "O fator X amplifica extremos (mais upside que downside)":
                    st.success("""
                    ✅ **Correto!** O efeito de X é maior nos extremos superiores (P90)
                    do que nos inferiores (P10). Isso sugere que X amplifica ganhos 
                    mais do que perdas — assimetria positiva.
                    """)
                else:
                    st.error("""
                    ❌ O efeito de X varia: 0.5 no P10, 1.0 na média, 1.5 no P90.
                    Isso indica que X tem mais impacto nos bons cenários do que nos ruins.
                    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa regressão quantílica para entender riscos assimétricos
    - Não confia apenas na média quando a distribuição é heterocedástica
    """)


def render_section_S6():
    """S6: Resumo Executivo e Ponte para o Próximo Módulo"""
    st.header("📋 Resumo Executivo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### O que Aprendemos sobre Extensões do CLRM
        
        ✅ **Regressão Múltipla:**
        - Permite incluir múltiplos drivers no mesmo modelo
        - Coeficientes são efeitos parciais (*ceteris paribus*)
        - Forma matricial y = Xβ + u é a base computacional
        
        ✅ **Teste F:**
        - Testa se um grupo de variáveis é conjuntamente significativo
        - Compara modelos restritos vs irrestritos
        - F = t² quando há apenas uma restrição
        
        ✅ **R² vs R² Ajustado:**
        - R² sempre aumenta com mais variáveis (mesmo inúteis)
        - R² ajustado penaliza complexidade desnecessária
        - Monitore ambos para evitar overfitting
        
        ✅ **Variáveis Dummy:**
        - Permitem incluir categorias qualitativas
        - Deslocam o intercepto (efeito aditivo)
        - Sempre omita uma categoria de referência
        
        ✅ **Regressão Quantílica:**
        - Estima efeitos em diferentes pontos da distribuição
        - Útil para análise de risco e extremos
        - Revela heterogeneidade que a média esconde
        """)
    
    with col2:
        st.markdown("### Checklist do Gestor")
        
        st.markdown("""
        📋 **Antes de modelar:**
        - [ ] Quais variáveis fazem sentido teórico?
        - [ ] Há categorias que precisam de dummies?
        
        📋 **Durante a análise:**
        - [ ] Coeficientes têm sinal esperado?
        - [ ] R² ajustado melhora com novas variáveis?
        - [ ] Teste F confirma significância conjunta?
        
        📋 **Para risco:**
        - [ ] A média é suficiente ou preciso ver quantis?
        - [ ] Há heterogeneidade nos efeitos?
        """)
    
    st.markdown("---")
    
    st.subheader("🔜 Próximo Módulo: Diagnósticos e Suposições")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Heterocedasticidade:**
        - Variância não constante dos erros
        - Testes: Breusch-Pagan, White
        - Correção: erros robustos
        """)
    
    with col2:
        st.markdown("""
        **Autocorrelação:**
        - Erros correlacionados no tempo
        - Teste: Durbin-Watson
        - Comum em séries temporais
        """)
    
    with col3:
        st.markdown("""
        **Multicolinearidade:**
        - Variáveis explicativas correlacionadas
        - Diagnóstico: VIF
        - Inflaciona erros padrão
        """)
    
    st.success("""
    🎓 **Mensagem final:** Com múltiplas variáveis, a interpretação fica mais rica 
    mas também mais complexa. Sempre questione: "Este coeficiente mede o que eu 
    quero, controlando pelo que preciso controlar?"
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa regressão múltipla como ferramenta padrão de análise
    - Conhece os limites: overfitting, significância espúria, necessidade de diagnósticos
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("📊 Módulo 3: Desenvolvimentos do CLRM")
    st.markdown("**Laboratório de Econometria** | Regressão Múltipla, Teste F, Dummies e Quantis")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Generalizar para **regressão múltipla** com interpretação ceteris paribus
        - Ensinar **Teste F** para hipóteses conjuntas
        - Discutir **R² vs R² ajustado** e prevenção de overfitting
        - Introduzir **variáveis dummy** e modelos hedônicos
        - Apresentar **regressão quantílica** para análise além da média
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "📊 Regressão Múltipla",
        "S2": "🧪 Teste F",
        "S3": "📈 R² e Seleção de Modelos",
        "S4": "🏠 Dummies e Hedônicos",
        "S5": "📊 Regressão Quantílica",
        "S6": "📋 Resumo e Próximos Passos"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Dica:** Este módulo expande o CLRM 
    com ferramentas essenciais para análise aplicada.
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


# =============================================================================
# EXECUÇÃO STANDALONE (para testes)
# =============================================================================

if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Módulo 3: Desenvolvimentos do CLRM",
            page_icon="📊",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()