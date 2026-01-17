"""
Laboratório de Econometria - Module 1: Statistics Review
Aplicativo educacional interativo para revisão de estatística aplicada a negócios.
Público-alvo: alunos de MBA com perfis quantitativos heterogêneos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# FUNÇÕES AUXILIARES PARA GERAÇÃO DE DADOS
# =============================================================================

@st.cache_data
def make_time_series(n_periods: int = 24, trend: float = 0.5, volatility: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Gera série temporal sintética com tendência e volatilidade ajustáveis."""
    np.random.seed(seed)
    dates = pd.date_range(start='2022-01-01', periods=n_periods, freq='M')
    trend_component = np.arange(n_periods) * trend
    noise = np.random.normal(0, volatility, n_periods)
    values = 100 + trend_component + noise.cumsum()
    return pd.DataFrame({'Data': dates, 'Valor': values})


@st.cache_data
def make_pooled(n_obs: int = 50, seed: int = 42) -> pd.DataFrame:
    """Gera dados pooled (cross-section) sintéticos."""
    np.random.seed(seed)
    empresas = [f'Empresa_{i+1}' for i in range(n_obs)]
    receita = np.random.lognormal(mean=4, sigma=0.5, size=n_obs)
    lucro = receita * np.random.uniform(0.05, 0.25, n_obs)
    setor = np.random.choice(['Varejo', 'Indústria', 'Serviços', 'Tech'], n_obs)
    return pd.DataFrame({'Empresa': empresas, 'Receita_MM': receita, 'Lucro_MM': lucro, 'Setor': setor})


@st.cache_data
def make_panel(n_entities: int = 4, n_periods: int = 8, seed: int = 42) -> pd.DataFrame:
    """Gera dados em painel (entidades × tempo)."""
    np.random.seed(seed)
    entities = [f'Unidade_{chr(65+i)}' for i in range(n_entities)]
    records = []
    for entity in entities:
        base = np.random.uniform(80, 120)
        for t in range(n_periods):
            value = base + t * np.random.uniform(0.5, 2) + np.random.normal(0, 5)
            records.append({'Entidade': entity, 'Período': t + 1, 'Valor': value})
    return pd.DataFrame(records)


@st.cache_data
def make_anscombe() -> dict:
    """Retorna o Quarteto de Anscombe como dicionário de DataFrames."""
    # Dados originais do Quarteto de Anscombe
    x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    
    x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    
    x3 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
    
    return {
        'I': pd.DataFrame({'x': x1, 'y': y1}),
        'II': pd.DataFrame({'x': x2, 'y': y2}),
        'III': pd.DataFrame({'x': x3, 'y': y3}),
        'IV': pd.DataFrame({'x': x4, 'y': y4})
    }


def make_salary_data(n: int = 100, outlier_value: float = 0, seed: int = 42) -> np.ndarray:
    """Gera dados de salários com outlier ajustável."""
    np.random.seed(seed)
    salaries = np.random.lognormal(mean=np.log(8000), sigma=0.4, size=n)
    if outlier_value > 0:
        salaries[-1] = outlier_value
    return salaries


def make_portfolio_returns(n: int = 252, sigma: float = 0.02, mu: float = 0.0005, seed: int = 42) -> np.ndarray:
    """Gera retornos diários simulados de carteira."""
    np.random.seed(seed)
    return np.random.normal(mu, sigma, n)


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: Por que estatística importa (negócios)"""
    st.header("📊 Por que Estatística Importa para Negócios")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        A estatística é a **linguagem da incerteza**. Em negócios, decisões são tomadas 
        sob informação imperfeita. A econometria nos ajuda a:
        
        - **Quantificar** o que sabemos e o que não sabemos
        - **Reduzir** a incerteza com dados e modelos
        - **Comunicar** riscos e oportunidades de forma precisa
        """)
        
        caso = st.selectbox(
            "Selecione um caso de uso:",
            ["Forecast de Vendas", "Gestão de Risco/Volatilidade", 
             "Avaliação de Performance", "Impacto de Política Interna"]
        )
        
        casos_desc = {
            "Forecast de Vendas": "Prever receita para planejamento de estoque e capacidade.",
            "Gestão de Risco/Volatilidade": "Dimensionar exposição a perdas e definir limites.",
            "Avaliação de Performance": "Separar sorte de competência na análise de resultados.",
            "Impacto de Política Interna": "Medir se uma mudança (preço, processo) teve efeito real."
        }
        st.info(f"**Aplicação:** {casos_desc[caso]}")
    
    with col2:
        st.subheader("Mini-Simulador: Incerteza e Previsão")
        
        valor_base = st.slider("Valor base esperado (R$ mil)", 100, 500, 200)
        incerteza = st.slider("Nível de incerteza (σ)", 10, 100, 30)
        
        np.random.seed(123)
        simulacoes = np.random.normal(valor_base, incerteza, 1000)
        
        p5, p50, p95 = np.percentile(simulacoes, [5, 50, 95])
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("P5 (pessimista)", f"R$ {p5:.0f} mil")
        col_m2.metric("P50 (mediana)", f"R$ {p50:.0f} mil")
        col_m3.metric("P95 (otimista)", f"R$ {p95:.0f} mil")
        
        fig = px.histogram(simulacoes, nbins=40, 
                          labels={'value': 'Valor (R$ mil)', 'count': 'Frequência'},
                          title="Distribuição de Cenários")
        fig.add_vline(x=p5, line_dash="dash", line_color="red", annotation_text="P5")
        fig.add_vline(x=p95, line_dash="dash", line_color="green", annotation_text="P95")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("💡 Pergunta Guiada"):
        st.markdown("**Qual o papel da Econometria na redução da incerteza?**")
        if st.button("Ver resposta", key="resp_s1"):
            st.success("""
            A Econometria usa dados históricos e teoria para **estreitar o intervalo de possibilidades**.
            Não elimina a incerteza, mas a **quantifica** e permite decisões mais informadas.
            Um gestor que conhece o intervalo P5-P95 pode dimensionar estoques, capital e contingências.
            """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Define metas realistas com intervalos de confiança
    - Dimensiona reservas e buffers para cenários adversos
    """)


def render_section_S2():
    """S2: Tipos de dados (qualitativo vs quantitativo)"""
    st.header("📋 Tipos de Dados: Qualitativo vs Quantitativo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        O **tipo de dado** determina quais análises são possíveis:
        
        | Tipo | Descrição | Exemplo |
        |------|-----------|---------|
        | **Nominal** | Categorias sem ordem | Setor, País, Cor |
        | **Ordinal** | Categorias com ordem | Rating (AAA>AA>A), Satisfação (1-5) |
        | **Discreta** | Números inteiros contáveis | Nº de funcionários, Nº de filiais |
        | **Contínua** | Números em escala contínua | Receita, Preço, Retorno % |
        """)
        
        st.markdown("""
        **Regra prática:**
        - **Qualitativo** (nominal/ordinal): segmentar, agrupar, comparar
        - **Quantitativo** (discreto/contínuo): prever, correlacionar, modelar
        """)
    
    with col2:
        st.subheader("🧪 Quiz Rápido")
        
        perguntas = [
            ("Código do setor econômico (CNAE)", "Nominal", 
             "Embora seja um número, representa uma categoria sem ordem."),
            ("Nota de crédito (AAA, AA, A, BBB...)", "Ordinal",
             "Categorias com ordem clara de qualidade."),
            ("Retorno anual do Ibovespa (%)", "Contínua",
             "Variável numérica que pode assumir qualquer valor real."),
        ]
        
        score = 0
        for i, (pergunta, resposta_correta, explicacao) in enumerate(perguntas):
            st.markdown(f"**{i+1}. Classifique:** *{pergunta}*")
            opcoes = ["Nominal", "Ordinal", "Discreta", "Contínua"]
            resposta = st.radio(f"Tipo:", opcoes, key=f"quiz_s2_{i}", horizontal=True)
            
            if st.button(f"Verificar {i+1}", key=f"check_s2_{i}"):
                if resposta == resposta_correta:
                    st.success(f"✅ Correto! {explicacao}")
                    score += 1
                else:
                    st.error(f"❌ Resposta: **{resposta_correta}**. {explicacao}")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Escolhe a análise correta para o tipo de dado disponível
    - Evita erros como calcular "média" de códigos de setor
    """)


def render_section_S3():
    """S3: Organização: time series, pooled, painel"""
    st.header("📁 Organização dos Dados: Série Temporal, Pooled, Painel")
    
    estrutura = st.radio(
        "Escolha a estrutura de dados:",
        ["Série Temporal (Time Series)", "Pooled (Cross-Section)", "Dados em Painel"],
        horizontal=True
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if estrutura == "Série Temporal (Time Series)":
            st.markdown("""
            **Uma entidade, vários períodos.**
            
            - **Uso:** Previsão, análise de tendência, sazonalidade
            - **Exemplo:** Vendas mensais de uma empresa
            - **Pergunta típica:** *"Qual será a receita do próximo trimestre?"*
            """)
            df = make_time_series(n_periods=24, trend=2, volatility=5)
            st.dataframe(df.head(8), use_container_width=True)
            
        elif estrutura == "Pooled (Cross-Section)":
            st.markdown("""
            **Várias entidades, um momento no tempo.**
            
            - **Uso:** Benchmarking, comparação, segmentação
            - **Exemplo:** Performance de várias empresas em 2023
            - **Pergunta típica:** *"Como nossa margem se compara ao setor?"*
            """)
            df = make_pooled(n_obs=20)
            st.dataframe(df.head(8), use_container_width=True)
            
        else:  # Painel
            st.markdown("""
            **Várias entidades × vários períodos.**
            
            - **Uso:** Inferência causal, controle de heterogeneidade
            - **Exemplo:** 50 lojas ao longo de 12 meses
            - **Pergunta típica:** *"A promoção aumentou vendas controlando por loja?"*
            """)
            df = make_panel(n_entities=4, n_periods=8)
            st.dataframe(df.head(12), use_container_width=True)
    
    with col2:
        st.subheader("Visualização")
        
        if estrutura == "Série Temporal (Time Series)":
            df = make_time_series(n_periods=24, trend=2, volatility=5)
            fig = px.line(df, x='Data', y='Valor', 
                         title="Série Temporal: Evolução no Tempo",
                         markers=True)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
        elif estrutura == "Pooled (Cross-Section)":
            df = make_pooled(n_obs=30)
            fig = px.scatter(df, x='Receita_MM', y='Lucro_MM', color='Setor',
                            title="Cross-Section: Comparação entre Entidades",
                            hover_data=['Empresa'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Painel
            df = make_panel(n_entities=4, n_periods=8)
            fig = px.line(df, x='Período', y='Valor', color='Entidade',
                         title="Painel: Múltiplas Entidades ao Longo do Tempo",
                         markers=True)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Identifica qual estrutura seus dados têm antes de escolher o modelo
    - Série temporal → ARIMA, suavização; Pooled → regressão cross-section; Painel → efeitos fixos/aleatórios
    """)


def render_section_S4():
    """S4: Média, mediana e moda (decisão)"""
    st.header("📏 Média, Mediana e Moda: Qual Usar?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Medidas de tendência central** resumem "onde está o centro" dos dados.
        Mas o "centro" depende do contexto e da presença de outliers.
        """)
        
        st.subheader("Controles")
        n_funcionarios = st.slider("Número de funcionários", 50, 200, 100)
        outlier = st.slider("Salário do CEO (outlier)", 0, 500000, 0, step=10000,
                           help="Adicione um salário extremo para ver o efeito")
        
        salarios = make_salary_data(n=n_funcionarios, outlier_value=outlier)
        
        media = np.mean(salarios)
        mediana = np.median(salarios)
        moda_bin = pd.cut(salarios, bins=20).value_counts().idxmax()
        moda_aprox = (moda_bin.left + moda_bin.right) / 2
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Média", f"R$ {media:,.0f}")
        col_m2.metric("Mediana", f"R$ {mediana:,.0f}")
        col_m3.metric("Moda (aprox)", f"R$ {moda_aprox:,.0f}")
        
        if outlier > 0:
            diff_pct = ((media - mediana) / mediana) * 100
            st.warning(f"⚠️ Com o outlier, a média está {diff_pct:.1f}% acima da mediana!")
    
    with col2:
        fig = px.histogram(salarios, nbins=30,
                          labels={'value': 'Salário (R$)', 'count': 'Frequência'},
                          title="Distribuição de Salários")
        fig.add_vline(x=media, line_dash="solid", line_color="red", 
                     annotation_text="Média", annotation_position="top")
        fig.add_vline(x=mediana, line_dash="dash", line_color="blue",
                     annotation_text="Mediana", annotation_position="top")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📖 Quando usar cada medida?"):
        st.markdown("""
        | Situação | Medida Recomendada | Exemplo |
        |----------|-------------------|---------|
        | Dados simétricos, sem outliers | **Média** | Notas de prova padronizada |
        | Dados assimétricos ou com outliers | **Mediana** | Salários, preços de imóveis |
        | Dados categóricos ou discretos | **Moda** | Tamanho de roupa mais vendido |
        | Decisão de remuneração | **Mediana** | Benchmark salarial de mercado |
        | Precificação | **Moda ou Mediana** | Preço mais comum aceito |
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa **mediana** para benchmarks salariais (evita distorção por executivos)
    - Reporta **média E mediana** quando há assimetria para dar contexto completo
    """)


def render_section_S5():
    """S5: Variância e dispersão (onde mora o risco)"""
    st.header("📉 Variância e Dispersão: Onde Mora o Risco")
    
    st.markdown("""
    Duas carteiras podem ter o **mesmo retorno médio**, mas riscos muito diferentes.
    A dispersão mede a **volatilidade** — quanto os resultados variam ao redor da média.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configure duas carteiras")
        
        st.markdown("**Carteira A (Conservadora)**")
        sigma_a = st.slider("Volatilidade A (σ %)", 0.5, 5.0, 1.0, 0.1)
        
        st.markdown("**Carteira B (Arrojada)**")
        sigma_b = st.slider("Volatilidade B (σ %)", 0.5, 5.0, 3.0, 0.1)
        
        # Mesmo retorno esperado
        mu = 0.05  # 5% ao ano, diário ~0.02%
        
        ret_a = make_portfolio_returns(n=252, sigma=sigma_a/100, mu=mu/252, seed=42)
        ret_b = make_portfolio_returns(n=252, sigma=sigma_b/100, mu=mu/252, seed=99)
        
        # Métricas
        st.markdown("---")
        st.markdown("**Métricas de Risco:**")
        
        metrics_df = pd.DataFrame({
            'Métrica': ['Retorno Médio Diário', 'Desvio Padrão', 'IQR', 'P5', 'P95'],
            'Carteira A': [
                f"{np.mean(ret_a)*100:.3f}%",
                f"{np.std(ret_a)*100:.2f}%",
                f"{(np.percentile(ret_a, 75) - np.percentile(ret_a, 25))*100:.2f}%",
                f"{np.percentile(ret_a, 5)*100:.2f}%",
                f"{np.percentile(ret_a, 95)*100:.2f}%"
            ],
            'Carteira B': [
                f"{np.mean(ret_b)*100:.3f}%",
                f"{np.std(ret_b)*100:.2f}%",
                f"{(np.percentile(ret_b, 75) - np.percentile(ret_b, 25))*100:.2f}%",
                f"{np.percentile(ret_b, 5)*100:.2f}%",
                f"{np.percentile(ret_b, 95)*100:.2f}%"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Histograma comparativo
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ret_a*100, name='Carteira A', opacity=0.7, nbinsx=40))
        fig.add_trace(go.Histogram(x=ret_b*100, name='Carteira B', opacity=0.7, nbinsx=40))
        fig.update_layout(
            title="Distribuição de Retornos Diários",
            xaxis_title="Retorno (%)",
            yaxis_title="Frequência",
            barmode='overlay',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Boxplot comparativo
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=ret_a*100, name='Carteira A'))
        fig2.add_trace(go.Box(y=ret_b*100, name='Carteira B'))
        fig2.update_layout(
            title="Boxplot: Dispersão e Outliers",
            yaxis_title="Retorno (%)",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.info("""
    💡 **Mensagem-chave:** Risco não é só a média — a dispersão define a probabilidade 
    de resultados extremos. Uma carteira com maior σ pode ter o mesmo retorno esperado, 
    mas perdas (e ganhos) muito maiores.
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Escolhe carteiras/projetos considerando retorno E risco (trade-off)
    - Dimensiona capital de reserva com base em P5 (cenário pessimista)
    """)


def render_section_S6():
    """S6: Quarteto de Anscombe (visualização obrigatória)"""
    st.header("🎨 Quarteto de Anscombe: Nunca Confie Apenas em Estatísticas")
    
    st.markdown("""
    O Quarteto de Anscombe demonstra que **quatro datasets completamente diferentes** 
    podem ter estatísticas-resumo praticamente idênticas. A lição? **Sempre visualize seus dados.**
    """)
    
    anscombe = make_anscombe()
    
    # Calcular estatísticas
    stats = {}
    for key, df in anscombe.items():
        stats[key] = {
            'Média X': df['x'].mean(),
            'Média Y': df['y'].mean(),
            'Var X': df['x'].var(),
            'Var Y': df['y'].var(),
            'Corr': df['x'].corr(df['y'])
        }
    
    # Mostrar tabela de estatísticas
    stats_df = pd.DataFrame(stats).T
    stats_df = stats_df.round(2)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Estatísticas (quase idênticas!)")
        st.dataframe(stats_df, use_container_width=True)
        
        mostrar_regressao = st.checkbox("Mostrar linha de regressão", value=True)
        
        st.markdown("""
        **Observe:** Todos os quatro conjuntos têm:
        - Mesma média de X (~9)
        - Mesma média de Y (~7.5)
        - Mesma variância
        - Mesma correlação (~0.82)
        - Mesma reta de regressão!
        """)
    
    with col2:
        st.subheader("Visualização (completamente diferentes!)")
        
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=['Dataset I', 'Dataset II', 'Dataset III', 'Dataset IV'])
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
        
        for (key, df), (row, col), color in zip(anscombe.items(), positions, colors):
            fig.add_trace(
                go.Scatter(x=df['x'], y=df['y'], mode='markers',
                          marker=dict(size=10, color=color),
                          name=f'Dataset {key}', showlegend=False),
                row=row, col=col
            )
            
            if mostrar_regressao:
                # Regressão linear
                slope, intercept = np.polyfit(df['x'], df['y'], 1)
                x_line = np.array([df['x'].min(), df['x'].max()])
                y_line = slope * x_line + intercept
                fig.add_trace(
                    go.Scatter(x=x_line, y=y_line, mode='lines',
                              line=dict(color='red', dash='dash'),
                              showlegend=False),
                    row=row, col=col
                )
        
        fig.update_layout(height=500)
        fig.update_xaxes(title_text="X")
        fig.update_yaxes(title_text="Y")
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("✅ Checklist: Lições para Executivos"):
        st.markdown("""
        - [ ] **Sempre visualize** antes de confiar em estatísticas-resumo
        - [ ] **Busque outliers** — eles podem distorcer médias e correlações
        - [ ] **Desconfie de resultados "bonitos"** — R² alto não garante relação válida
        - [ ] **Entenda o processo gerador** — dados podem ter estruturas ocultas
        - [ ] **Peça o gráfico** quando alguém apresentar apenas números
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Exige visualizações em toda análise de dados apresentada
    - Questiona relatórios que mostram apenas estatísticas-resumo
    """)


def render_section_S7():
    """S7: Correlação ≠ causalidade (alerta executivo)"""
    st.header("⚠️ Correlação ≠ Causalidade: O Alerta Executivo")
    
    st.markdown("""
    Duas variáveis podem estar correlacionadas por três motivos:
    1. **X causa Y** (relação causal direta)
    2. **Y causa X** (causalidade reversa)
    3. **Z causa ambos** (variável omitida / confundidor)
    
    A correlação sozinha não distingue esses casos!
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Simulação: Correlação Espúria")
        
        st.markdown("""
        **Cenário:** Marketing e Vendas parecem correlacionados, 
        mas ambos são causados pela **Sazonalidade** (confundidor).
        """)
        
        intensidade_confounder = st.slider(
            "Intensidade do confundidor (sazonalidade)", 
            0.0, 1.0, 0.8, 0.1,
            help="Quanto maior, mais a sazonalidade afeta marketing E vendas"
        )
        
        n = 100
        np.random.seed(42)
        
        # Confundidor: sazonalidade (ciclo)
        sazonalidade = np.sin(np.linspace(0, 4*np.pi, n))
        
        # Marketing afetado pela sazonalidade + ruído
        marketing = 50 + 20 * sazonalidade * intensidade_confounder + np.random.normal(0, 5, n)
        
        # Vendas afetadas pela sazonalidade + ruído (NÃO pelo marketing diretamente)
        vendas = 100 + 30 * sazonalidade * intensidade_confounder + np.random.normal(0, 8, n)
        
        corr = np.corrcoef(marketing, vendas)[0, 1]
        
        st.metric("Correlação Marketing × Vendas", f"{corr:.2f}")
        
        if corr > 0.7:
            st.error("🚨 Alta correlação! Mas é causal ou espúria?")
        elif corr > 0.4:
            st.warning("⚠️ Correlação moderada — investigar confundidores")
        else:
            st.success("✅ Correlação fraca após controlar sazonalidade")
    
    with col2:
        df_sim = pd.DataFrame({
            'Marketing': marketing,
            'Vendas': vendas,
            'Sazonalidade': sazonalidade
        })
        
        fig = px.scatter(df_sim, x='Marketing', y='Vendas', 
                        color='Sazonalidade',
                        title="Marketing vs Vendas (cor = sazonalidade)",
                        color_continuous_scale='RdYlBu')
        
        # Linha de tendência
        slope, intercept = np.polyfit(marketing, vendas, 1)
        x_line = np.array([marketing.min(), marketing.max()])
        y_line = slope * x_line + intercept
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='Tendência'))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📖 O que posso concluir com correlação?"):
        st.markdown("""
        | Com Correlação | Precisa de Identificação Causal |
        |----------------|--------------------------------|
        | Existe associação entre X e Y | X causa Y |
        | Prever Y dado X (se relação estável) | Aumentar X aumentará Y |
        | Detectar padrões e anomalias | Atribuir efeito de política/intervenção |
        
        **Para identificação causal, precisamos de:**
        - Experimentos aleatorizados (A/B test)
        - Variáveis instrumentais
        - Diferenças-em-diferenças
        - Regression discontinuity
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Questiona "aumentar marketing aumenta vendas?" antes de decidir orçamento
    - Exige experimentos controlados para decisões de grande impacto
    """)


def render_section_S8():
    """S8: Distribuições e normalidade (caudas e assimetria)"""
    st.header("📊 Distribuições: Normal, Caudas Pesadas e Assimetria")
    
    st.markdown("""
    A escolha da distribuição afeta dramaticamente a avaliação de risco.
    **Eventos extremos** (caudas) podem ser subestimados pela distribuição normal.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📈 Histograma", "📦 Boxplot", "📋 Percentis"])
    
    dist_tipo = st.selectbox(
        "Escolha a distribuição:",
        ["Normal", "Log-Normal (assimétrica)", "t-Student (caudas pesadas)"]
    )
    
    n = 2000
    np.random.seed(42)
    
    if dist_tipo == "Normal":
        dados = np.random.normal(0, 1, n)
        info = "Simétrica, caudas leves. Assume que eventos extremos são muito raros."
    elif dist_tipo == "Log-Normal (assimétrica)":
        dados = np.random.lognormal(0, 0.5, n)
        dados = (dados - dados.mean()) / dados.std()  # Padronizar para comparação
        info = "Assimétrica à direita. Comum em retornos, preços, tempos de espera."
    else:  # t-Student
        dados = np.random.standard_t(df=3, size=n)
        dados = dados / dados.std()  # Padronizar
        info = "Simétrica, mas com caudas mais pesadas. Eventos extremos são mais frequentes."
    
    st.info(f"💡 **{dist_tipo}:** {info}")
    
    with tab1:
        fig = px.histogram(dados, nbins=60, 
                          title=f"Distribuição {dist_tipo}",
                          labels={'value': 'Valor', 'count': 'Frequência'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.box(y=dados, title=f"Boxplot - {dist_tipo}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        percentis = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        valores = [np.percentile(dados, p) for p in percentis]
        
        df_perc = pd.DataFrame({
            'Percentil': [f'P{p}' for p in percentis],
            'Valor': valores
        })
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(df_perc, use_container_width=True, hide_index=True)
        with col2:
            fig = px.bar(df_perc, x='Percentil', y='Valor',
                        title="Valores por Percentil")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🔥 Por que eventos extremos importam em risco?"):
        st.markdown("""
        - **VaR (Value at Risk)** depende da cauda esquerda da distribuição
        - Se assumimos normal quando a real é t-Student, **subestimamos perdas extremas**
        - Crises financeiras são "caudas" — a normal as trata como quase impossíveis
        - **Regra prática:** Para risco, use distribuições com caudas pesadas ou simule cenários extremos
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Não assume normalidade para cálculo de risco sem testar
    - Usa stress tests com cenários extremos além do modelo base
    """)


def render_section_S9():
    """S9: Amostra, população e vieses"""
    st.header("🎯 Amostra, População e Vieses de Seleção")
    
    st.markdown("""
    Se sua amostra não representa a população, suas conclusões serão **enviesadas**.
    O **viés de seleção** ocorre quando certas observações têm maior probabilidade de entrar na amostra.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Simulação: Viés de Sobrevivência")
        
        st.markdown("""
        **Cenário:** Você analisa apenas fundos que "sobreviveram" (existem hoje).
        Os fundos que faliram não estão na amostra!
        """)
        
        n_total = 500
        np.random.seed(42)
        
        # População: retornos de todos os fundos (incluindo os que faliram)
        retornos_populacao = np.random.normal(0.02, 0.15, n_total)  # Média 2%, vol 15%
        
        # Fundos que faliram: retornos muito negativos
        faliu = retornos_populacao < -0.20  # Fundos com perda > 20% fecharam
        
        pct_sobreviventes = st.slider(
            "% mínimo de sobreviventes para incluir na amostra",
            0, 100, 80, 5
        )
        
        # Amostra: apenas sobreviventes
        limiar = np.percentile(retornos_populacao, 100 - pct_sobreviventes)
        sobreviventes = retornos_populacao[retornos_populacao > limiar]
        
        media_pop = np.mean(retornos_populacao)
        media_amostra = np.mean(sobreviventes)
        vies = media_amostra - media_pop
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Média População", f"{media_pop*100:.1f}%")
        col_m2.metric("Média Amostra", f"{media_amostra*100:.1f}%", 
                     delta=f"{vies*100:+.1f}%")
        col_m3.metric("Viés", f"{vies*100:+.1f}%")
        
        if abs(vies) > 0.02:
            st.error(f"🚨 Viés significativo de {vies*100:.1f}%! A amostra superestima retornos.")
    
    with col2:
        # Visualização
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=retornos_populacao * 100, 
            name='População (todos)',
            opacity=0.5,
            nbinsx=40
        ))
        
        fig.add_trace(go.Histogram(
            x=sobreviventes * 100,
            name='Amostra (sobreviventes)',
            opacity=0.7,
            nbinsx=40
        ))
        
        fig.add_vline(x=media_pop*100, line_dash="solid", line_color="blue",
                     annotation_text="Média Pop")
        fig.add_vline(x=media_amostra*100, line_dash="dash", line_color="red",
                     annotation_text="Média Amostra")
        
        fig.update_layout(
            title="Distribuição: População vs Amostra Enviesada",
            xaxis_title="Retorno (%)",
            yaxis_title="Frequência",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("✅ Checklist: Diagnosticando Vieses em Dados Corporativos"):
        st.markdown("""
        Faça estas perguntas antes de confiar em qualquer análise:
        
        1. **Quem está faltando?**
           - Clientes que cancelaram estão na base?
           - Produtos descontinuados foram excluídos?
           - Funcionários demitidos aparecem nos dados de performance?
        
        2. **Como os dados foram coletados?**
           - Pesquisa foi respondida só por quem quis?
           - Há incentivo para certos grupos responderem?
        
        3. **O período é representativo?**
           - Dados de vendas incluem meses atípicos (Black Friday, COVID)?
           - Série histórica sobreviveu a mudanças metodológicas?
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Pergunta "quem está faltando nesta análise?" antes de decidir
    - Exige dados de "não-clientes" e "desistentes" em estudos de satisfação
    """)


def render_section_S10():
    """S10: Estatística como linguagem dos modelos (ponte para econometria)"""
    st.header("🌉 Estatística como Linguagem dos Modelos")
    
    st.markdown("""
    Estatística é o **vocabulário** que usaremos em todo o curso. 
    Ela conecta dados brutos à tomada de decisão através de um fluxo lógico.
    """)
    
    # Mapa conceitual usando Mermaid-style com Plotly
    st.subheader("Fluxo: De Dados à Decisão")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Criar visualização do fluxo
        fig = go.Figure()
        
        # Posições dos nós
        nodes = {
            'Dados': (0, 2),
            'Descrição': (1, 2),
            'Incerteza': (2, 2),
            'Modelo': (3, 2),
            'Decisão': (4, 2)
        }
        
        # Cores
        colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
        
        # Adicionar nós
        for i, (nome, (x, y)) in enumerate(nodes.items()):
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=60, color=colors[i]),
                text=[nome],
                textposition='middle center',
                textfont=dict(color='white', size=12),
                showlegend=False
            ))
        
        # Adicionar setas (linhas)
        for i in range(len(nodes) - 1):
            x_vals = list(nodes.values())
            fig.add_trace(go.Scatter(
                x=[x_vals[i][0] + 0.15, x_vals[i+1][0] - 0.15],
                y=[2, 2],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        # Adicionar descrições abaixo
        descricoes = [
            'Coletar\nOrganizar',
            'Média, Variância\nDistribuição',
            'Quantificar\no que não sabemos',
            'Regressão\nInferência',
            'Ação\ncom fundamento'
        ]
        
        for i, (nome, (x, y)) in enumerate(nodes.items()):
            fig.add_trace(go.Scatter(
                x=[x], y=[1.3],
                mode='text',
                text=[descricoes[i]],
                textfont=dict(size=10, color='gray'),
                showlegend=False
            ))
        
        fig.update_layout(
            height=250,
            xaxis=dict(visible=False, range=[-0.5, 4.5]),
            yaxis=dict(visible=False, range=[0.8, 2.5]),
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Cada etapa usa estatística:**
        
        1. **Dados** → Tipos, organização
        2. **Descrição** → Média, mediana, variância
        3. **Incerteza** → Distribuições, intervalos
        4. **Modelo** → Regressão, inferência
        5. **Decisão** → Ação fundamentada
        """)
    
    st.markdown("---")
    
    # Resumo do módulo
    st.subheader("📝 Resumo do Módulo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **O que aprendemos:**
        
        - Estatística reduz incerteza e fundamenta decisões
        - Tipo de dado determina análise possível
        - Estrutura (TS/pooled/painel) define o modelo
        - Média vs mediana: robustez importa
        - Dispersão é risco, não só média
        - Visualização é obrigatória (Anscombe)
        - Correlação ≠ causalidade
        - Distribuições afetam avaliação de risco
        - Viés de seleção distorce conclusões
        """)
    
    with col2:
        st.markdown("""
        **Próximo módulo: CLRM**
        
        No próximo módulo, construiremos nosso primeiro modelo:
        o **Modelo Clássico de Regressão Linear (CLRM)**.
        
        Usaremos toda a linguagem estatística aprendida aqui para:
        - Estimar relações entre variáveis
        - Testar hipóteses
        - Fazer previsões com intervalos de confiança
        """)
    
    st.success("""
    🎓 **Fluência em dados** significa pensar estatisticamente: 
    questionar resumos, visualizar distribuições, identificar vieses, 
    e distinguir correlação de causalidade. Você está pronto para a econometria!
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Desenvolve "intuição quantitativa" para questionar análises
    - Exige rigor metodológico nas decisões baseadas em dados
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("📊 Módulo 1: Revisão de Estatística")
    st.markdown("**Laboratório de Econometria** | Fundamentos para Análise de Dados em Negócios")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Revisar estatística essencial para econometria, priorizando **interpretação e decisão**
        - Mostrar como escolhas de resumo/visualização mudam conclusões
        - Ensinar a **pensar com dados**: tipos, organização, medidas, vieses
        - Conectar estatística à **redução de incerteza** e tomada de decisão gerencial
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "📊 Por que Estatística Importa",
        "S2": "📋 Tipos de Dados",
        "S3": "📁 Organização dos Dados",
        "S4": "📏 Média, Mediana e Moda",
        "S5": "📉 Variância e Dispersão",
        "S6": "🎨 Quarteto de Anscombe",
        "S7": "⚠️ Correlação ≠ Causalidade",
        "S8": "📊 Distribuições e Normalidade",
        "S9": "🎯 Amostra e Vieses",
        "S10": "🌉 Ponte para Econometria"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Dica:** Explore os controles interativos 
    em cada seção para construir intuição.
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
    elif secao_selecionada == "S10":
        render_section_S10()


# =============================================================================
# EXECUÇÃO STANDALONE (para testes)
# =============================================================================

if __name__ == "__main__":
    # Configuração da página (apenas quando executado diretamente)
    # Quando importado por econometrics_lab.py, esta configuração NÃO é executada
    try:
        st.set_page_config(
            page_title="Módulo 1: Revisão de Estatística",
            page_icon="📊",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        # Já foi configurado pelo app principal
        pass
    render()