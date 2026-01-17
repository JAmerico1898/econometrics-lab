"""
Laboratório de Econometria - Module 5: Causality and Identification
Aplicativo educacional interativo para causalidade e identificação em negócios.
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
def make_spurious_corr_data(n: int = 100, confounder_effect: float = 0.8, seed: int = 42) -> pd.DataFrame:
    """Gera dados com correlação espúria via confundidor."""
    np.random.seed(seed)
    
    # Confundidor: sazonalidade/tendência econômica
    confounder = np.random.normal(0, 1, n)
    
    # Marketing afetado pelo confundidor (empresas investem mais em épocas boas)
    marketing = 50 + 15 * confounder_effect * confounder + np.random.normal(0, 5, n)
    
    # Vendas afetadas pelo confundidor (vendas sobem em épocas boas)
    # NÃO há efeito direto do marketing nesta simulação
    vendas = 100 + 25 * confounder_effect * confounder + np.random.normal(0, 10, n)
    
    return pd.DataFrame({
        'Marketing': marketing,
        'Vendas': vendas,
        'Economia': confounder  # O confundidor oculto
    })


@st.cache_data
def make_counterfactual_sim(n_periods: int = 24, treatment_period: int = 12,
                            true_effect: float = 20, seed: int = 42) -> pd.DataFrame:
    """Simula cenário com e sem tratamento para ilustrar contrafactual."""
    np.random.seed(seed)
    
    t = np.arange(1, n_periods + 1)
    
    # Tendência base
    trend = 2 * t + np.random.normal(0, 3, n_periods)
    
    # Cenário sem tratamento (contrafactual)
    y_counterfactual = 50 + trend
    
    # Cenário com tratamento
    y_observed = y_counterfactual.copy()
    y_observed[treatment_period:] += true_effect
    
    # Adicionar ruído
    y_observed += np.random.normal(0, 2, n_periods)
    y_counterfactual += np.random.normal(0, 2, n_periods)
    
    treatment = np.array(['Antes'] * treatment_period + ['Depois'] * (n_periods - treatment_period))
    
    return pd.DataFrame({
        't': t,
        'Observado': y_observed,
        'Contrafactual': y_counterfactual,
        'Periodo': treatment
    })


@st.cache_data
def make_omitted_var_data(n: int = 200, omitted_effect: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Gera dados com viés de variável omitida."""
    np.random.seed(seed)
    
    # Variável omitida: experiência do funcionário
    experiencia = np.random.uniform(1, 20, n)
    
    # Treinamento (correlacionado com experiência - mais experientes fazem mais treinamento)
    treinamento = 10 + 0.5 * experiencia * omitted_effect + np.random.normal(0, 3, n)
    treinamento = np.clip(treinamento, 0, None)
    
    # Produtividade depende de experiência E treinamento
    # Efeito verdadeiro do treinamento: 2
    produtividade = 30 + 2 * treinamento + 3 * experiencia + np.random.normal(0, 5, n)
    
    return pd.DataFrame({
        'Treinamento': treinamento,
        'Produtividade': produtividade,
        'Experiencia': experiencia
    })


@st.cache_data
def make_reverse_causality_data(n: int = 100, reverse_effect: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Gera dados com causalidade reversa."""
    np.random.seed(seed)
    
    # Lucro base
    lucro_base = np.random.normal(100, 20, n)
    
    # Investimento depende do lucro (causalidade reversa)
    investimento = 20 + 0.3 * lucro_base * reverse_effect + np.random.normal(0, 5, n)
    
    # Lucro também é afetado por investimento (efeito verdadeiro: 0.5)
    lucro = lucro_base + 0.5 * investimento + np.random.normal(0, 10, n)
    
    return pd.DataFrame({
        'Investimento': investimento,
        'Lucro': lucro
    })


@st.cache_data
def make_selection_bias_data(n: int = 200, selection_intensity: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Gera dados com viés de seleção."""
    np.random.seed(seed)
    
    # Habilidade latente (não observada)
    habilidade = np.random.normal(50, 10, n)
    
    # Probabilidade de receber treinamento aumenta com habilidade
    prob_treinamento = 1 / (1 + np.exp(-(habilidade - 50) * selection_intensity / 10))
    treinamento = np.random.binomial(1, prob_treinamento)
    
    # Produtividade depende de habilidade e treinamento (efeito verdadeiro: 10)
    produtividade = 30 + 0.8 * habilidade + 10 * treinamento + np.random.normal(0, 5, n)
    
    return pd.DataFrame({
        'Treinamento': treinamento,
        'Produtividade': produtividade,
        'Habilidade': habilidade
    })


@st.cache_data
def make_ab_test_sim(n_total: int = 1000, true_effect: float = 5, seed: int = 42) -> pd.DataFrame:
    """Simula um teste A/B com atribuição aleatória."""
    np.random.seed(seed)
    
    # Atribuição aleatória
    tratamento = np.random.binomial(1, 0.5, n_total)
    
    # Características de base (balanceadas por aleatorização)
    idade = np.random.normal(35, 10, n_total)
    renda = np.random.lognormal(10, 0.5, n_total)
    
    # Conversão: depende das características + efeito do tratamento
    prob_base = 0.05 + 0.001 * (idade - 35) + 0.00001 * (renda - np.exp(10))
    prob_conversao = prob_base + true_effect / 100 * tratamento
    prob_conversao = np.clip(prob_conversao, 0.01, 0.99)
    
    conversao = np.random.binomial(1, prob_conversao)
    
    return pd.DataFrame({
        'Tratamento': tratamento,
        'Conversao': conversao,
        'Idade': idade,
        'Renda': renda,
        'Grupo': np.where(tratamento == 1, 'Tratamento', 'Controle')
    })


@st.cache_data
def make_quasi_experiment_data(n: int = 200, cutoff: float = 50, effect: float = 15, seed: int = 42) -> pd.DataFrame:
    """Gera dados para quase-experimento (regression discontinuity)."""
    np.random.seed(seed)
    
    # Score que determina elegibilidade (ex.: nota, renda, idade)
    score = np.random.uniform(20, 80, n)
    
    # Tratamento: elegível se score >= cutoff
    tratamento = (score >= cutoff).astype(int)
    
    # Resultado: depende do score + efeito do tratamento no ponto de corte
    resultado = 30 + 0.5 * score + effect * tratamento + np.random.normal(0, 5, n)
    
    return pd.DataFrame({
        'Score': score,
        'Tratamento': tratamento,
        'Resultado': resultado,
        'Grupo': np.where(tratamento == 1, 'Elegível', 'Não Elegível')
    })


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: Correlação não é Causalidade"""
    st.header("🔗 Correlação não é Causalidade")
    
    st.markdown("""
    A frase mais importante em análise de dados: **correlação não implica causalidade**.
    Duas variáveis podem estar fortemente associadas sem que uma cause a outra.
    """)
    
    tab1, tab2 = st.tabs(["📊 Simulação", "💼 Exemplos de Negócio"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("O Confundidor Oculto")
            
            confounder_effect = st.slider(
                "Intensidade do confundidor (economia)",
                0.0, 1.0, 0.8, 0.1,
                help="Quanto a economia afeta marketing E vendas"
            )
            
            st.markdown("""
            **Simulação:**
            - Marketing e Vendas são **ambos** afetados pela economia
            - NÃO há efeito direto de Marketing sobre Vendas
            - Mas a correlação parece alta!
            """)
            
            df = make_spurious_corr_data(n=100, confounder_effect=confounder_effect)
            corr = df['Marketing'].corr(df['Vendas'])
            
            st.metric("Correlação Marketing × Vendas", f"{corr:.2f}")
            
            if corr > 0.6:
                st.error("🚨 Alta correlação! Mas é causal? NÃO nesta simulação.")
            else:
                st.info("Correlação mais baixa porque o confundidor é fraco.")
        
        with col2:
            fig = px.scatter(df, x='Marketing', y='Vendas', color='Economia',
                            color_continuous_scale='RdYlGn',
                            title="Marketing vs Vendas (cor = estado da economia)")
            
            # Linha de tendência
            z = np.polyfit(df['Marketing'], df['Vendas'], 1)
            x_line = np.linspace(df['Marketing'].min(), df['Marketing'].max(), 50)
            fig.add_trace(go.Scatter(x=x_line, y=z[0]*x_line + z[1],
                                    mode='lines', line=dict(color='red', dash='dash'),
                                    name=f'Tendência (r={corr:.2f})'))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Exemplos Clássicos em Negócios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📈 Marketing vs Vendas**
            - Correlação alta entre gasto em marketing e vendas
            - Mas: empresas aumentam marketing quando a economia vai bem
            - E vendas também sobem com a economia
            - **Confundidor:** Ciclo econômico
            
            **🎓 Treinamento vs Produtividade**
            - Funcionários treinados são mais produtivos
            - Mas: funcionários motivados buscam mais treinamento
            - E funcionários motivados já são mais produtivos
            - **Confundidor:** Motivação intrínseca
            """)
        
        with col2:
            st.markdown("""
            **🏦 Crédito vs Inadimplência**
            - Clientes com mais crédito têm menos inadimplência
            - Mas: bancos dão mais crédito a quem tem renda alta
            - E quem tem renda alta paga mais em dia
            - **Confundidor:** Renda
            
            **💊 Medicamento vs Recuperação**
            - Pacientes que tomam o remédio se recuperam mais
            - Mas: médicos prescrevem para casos menos graves
            - E casos menos graves se recuperam mais rápido
            - **Confundidor:** Gravidade do caso
            """)
        
        st.warning("""
        ⚠️ **Alerta executivo:** Antes de decidir "vamos aumentar X porque está 
        correlacionado com Y", pergunte: "Existe um terceiro fator que causa ambos?"
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Nunca assume causalidade apenas por correlação
    - Pergunta: "O que mais poderia explicar essa associação?"
    """)


def render_section_S2():
    """S2: O Problema do Contrafactual"""
    st.header("🔮 O Problema do Contrafactual")
    
    st.markdown("""
    A pergunta causal fundamental é: **"O que teria acontecido se eu NÃO tivesse feito X?"**
    
    Esse cenário alternativo — que nunca observamos — é o **contrafactual**.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("O Desafio")
        
        st.markdown("""
        **Exemplo:** Você lançou uma campanha de marketing no mês 12.
        As vendas subiram 20% nos meses seguintes.
        
        **Pergunta causal:** A campanha causou o aumento?
        
        **O problema:** Não sabemos o que teria acontecido SEM a campanha.
        Talvez as vendas subissem de qualquer forma (sazonalidade, tendência).
        """)
        
        st.info("""
        💡 **Contrafactual:** O cenário hipotético onde tudo é igual, 
        exceto pela intervenção que queremos avaliar.
        """)
        
        true_effect = st.slider("Efeito verdadeiro da campanha", 0, 40, 20, 5)
        treatment_period = st.slider("Mês da campanha", 6, 18, 12)
    
    with col2:
        df = make_counterfactual_sim(n_periods=24, treatment_period=treatment_period,
                                     true_effect=true_effect)
        
        fig = go.Figure()
        
        # Linha observada
        fig.add_trace(go.Scatter(x=df['t'], y=df['Observado'],
                                mode='lines+markers', name='Observado (com campanha)',
                                line=dict(color='blue', width=2)))
        
        # Linha contrafactual
        fig.add_trace(go.Scatter(x=df['t'], y=df['Contrafactual'],
                                mode='lines', name='Contrafactual (sem campanha)',
                                line=dict(color='red', dash='dash', width=2)))
        
        # Linha vertical no tratamento
        fig.add_vline(x=treatment_period, line_dash="dot", line_color="green",
                     annotation_text="Campanha")
        
        # Área do efeito
        fig.add_trace(go.Scatter(
            x=list(df['t'][treatment_period:]) + list(df['t'][treatment_period:][::-1]),
            y=list(df['Observado'][treatment_period:]) + list(df['Contrafactual'][treatment_period:][::-1]),
            fill='toself',
            fillcolor='rgba(0,255,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Efeito Causal'
        ))
        
        fig.update_layout(
            title="Cenário Observado vs Contrafactual",
            xaxis_title="Mês",
            yaxis_title="Vendas",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calcular efeito estimado
        media_depois_obs = df.loc[df['t'] > treatment_period, 'Observado'].mean()
        media_depois_cf = df.loc[df['t'] > treatment_period, 'Contrafactual'].mean()
        efeito_estimado = media_depois_obs - media_depois_cf
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Efeito Verdadeiro", f"{true_effect:.0f}")
        col_m2.metric("Efeito Estimado", f"{efeito_estimado:.1f}")
    
    with st.expander("💡 Por que nunca vemos o efeito puro?"):
        st.markdown("""
        **O problema fundamental da inferência causal:**
        
        Para medir o efeito de X sobre Y, precisaríamos observar:
        1. Y quando X acontece (observado ✓)
        2. Y quando X NÃO acontece, tudo mais igual (contrafactual ✗)
        
        Mas só podemos observar UM dos cenários para cada unidade!
        
        **Soluções práticas:**
        - Usar grupos de controle para aproximar o contrafactual
        - Randomização para garantir comparabilidade
        - Métodos estatísticos para estimar o contrafactual
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Entende que medir impacto requer comparação com cenário alternativo
    - Planeja avaliações com grupos de controle desde o início
    """)


def render_section_S3():
    """S3: Identificação: Isolando o Efeito Real"""
    st.header("🎯 Identificação: Isolando o Efeito Causal")
    
    st.markdown("""
    **Identificação** é o processo de isolar o efeito causal de interesse,
    separando-o de outros fatores que poderiam explicar a associação observada.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("O que é Identificação?")
        
        st.markdown("""
        Em linguagem de negócios, identificação responde:
        
        > *"Posso afirmar com confiança que X causou Y, 
        > e não foi outra coisa?"*
        
        **Condições para identificação:**
        1. Variação exógena em X (não causada por Y ou por confundidores)
        2. Ausência de fatores omitidos que afetam X e Y
        3. Direção causal clara (X → Y, não Y → X)
        """)
        
        st.subheader("Mapa Causal")
        
        st.markdown("""
        ```
        ┌─────────────┐
        │ Confundidor │
        │  (oculto)   │
        └──────┬──────┘
               │
          ┌────┴────┐
          ▼         ▼
        ┌───┐     ┌───┐
        │ X │ ──? │ Y │
        └───┘     └───┘
        Decisão   Resultado
        
        Pergunta: A seta X → Y é real?
        Ou é X ← Confundidor → Y?
        ```
        """)
    
    with col2:
        st.subheader("✅ Checklist de Identificação")
        
        st.markdown("""
        **Antes de afirmar causalidade, verifique:**
        """)
        
        checks = [
            ("Existe variação em X que não foi causada por Y?", 
             "Ex.: mudança de política, experimento, choque externo"),
            ("Controlei todos os fatores que afetam X e Y?",
             "Ex.: características do cliente, época do ano, tendência"),
            ("A direção causal faz sentido teórico?",
             "Ex.: é mais provável que marketing cause vendas do que o contrário?"),
            ("Há grupo de comparação válido?",
             "Ex.: quem não recebeu a intervenção é comparável a quem recebeu?"),
            ("O timing faz sentido?",
             "Ex.: X aconteceu antes de Y mudar?")
        ]
        
        for pergunta, exemplo in checks:
            with st.expander(f"☐ {pergunta}"):
                st.caption(exemplo)
        
        st.warning("""
        ⚠️ **Se alguma resposta for "não" ou "não sei":**
        A identificação está comprometida. O "efeito" pode ser espúrio.
        """)
    
    # Quiz
    st.subheader("🧪 Quiz Rápido")
    
    st.markdown("""
    **Cenário:** Uma empresa descobriu que lojas com mais funcionários têm vendas maiores.
    O CEO quer contratar mais pessoas em todas as lojas.
    """)
    
    resposta = st.radio(
        "O que você diria?",
        ["Ótima ideia! Mais funcionários causam mais vendas.",
         "Cuidado! Pode haver causalidade reversa ou confundidores.",
         "Impossível saber qualquer coisa com dados observacionais."],
        key="quiz_s3"
    )
    
    if st.button("Ver análise", key="btn_s3"):
        if resposta == "Cuidado! Pode haver causalidade reversa ou confundidores.":
            st.success("""
            ✅ **Correto!** Problemas de identificação:
            
            1. **Causalidade reversa:** Lojas com mais vendas contratam mais
            2. **Confundidor:** Lojas em locais melhores têm mais vendas E mais funcionários
            3. **Seleção:** A empresa pode ter alocado mais funcionários para lojas com potencial
            
            Para identificar, precisaríamos de variação exógena (ex.: experimento com alocação aleatória).
            """)
        else:
            st.error("A correlação observada não garante que contratar mais funcionários aumentará vendas.")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Exige evidência de identificação antes de basear decisões em correlações
    - Pergunta: "O que garante que essa relação é causal?"
    """)


def render_section_S4():
    """S4: Principais Ameaças à Causalidade"""
    st.header("⚠️ Principais Ameaças à Inferência Causal")
    
    st.markdown("""
    Três ameaças clássicas podem fazer você confundir correlação com causalidade:
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📦 Variável Omitida", "🔄 Causalidade Reversa", 
                                       "🎯 Viés de Seleção", "🔬 Simulador"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Omissão de Variáveis")
            
            st.markdown("""
            **O problema:** Uma variável que você não incluiu afeta tanto X quanto Y.
            
            **Exemplo: Treinamento → Produtividade**
            - Você observa: mais treinamento → mais produtividade
            - Mas esqueceu: experiência afeta AMBOS
            - Funcionários experientes fazem mais treinamento E são mais produtivos
            """)
            
            omitted_effect = st.slider("Correlação experiência-treinamento", 0.0, 1.0, 0.5, 0.1,
                                       key="omit_slider")
            
            df = make_omitted_var_data(n=200, omitted_effect=omitted_effect)
            
            # Regressão sem controle
            corr_naive = np.corrcoef(df['Treinamento'], df['Produtividade'])[0, 1]
            beta_naive = np.polyfit(df['Treinamento'], df['Produtividade'], 1)[0]
            
            st.metric("β estimado (sem controle)", f"{beta_naive:.2f}",
                     delta=f"Viés: {beta_naive - 2:.2f}" if abs(beta_naive - 2) > 0.1 else "≈ verdadeiro")
            st.caption("Efeito verdadeiro: β = 2")
        
        with col2:
            fig = px.scatter(df, x='Treinamento', y='Produtividade', color='Experiencia',
                            color_continuous_scale='Viridis',
                            title="Treinamento vs Produtividade (cor = experiência)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Causalidade Reversa")
            
            st.markdown("""
            **O problema:** Y causa X, não o contrário (ou ambos se causam).
            
            **Exemplo: Investimento → Lucro**
            - Você observa: mais investimento → mais lucro
            - Mas também: mais lucro → mais investimento
            - Empresas lucrativas têm caixa para investir!
            """)
            
            reverse_effect = st.slider("Intensidade da causalidade reversa", 0.0, 1.0, 0.5, 0.1,
                                       key="reverse_slider")
            
            df = make_reverse_causality_data(n=100, reverse_effect=reverse_effect)
            
            corr = df['Investimento'].corr(df['Lucro'])
            beta = np.polyfit(df['Investimento'], df['Lucro'], 1)[0]
            
            st.metric("Correlação", f"{corr:.2f}")
            st.metric("β estimado", f"{beta:.2f}",
                     delta="Viés de simultaneidade" if reverse_effect > 0.3 else "")
            st.caption("Efeito verdadeiro Inv→Lucro: β = 0.5")
        
        with col2:
            fig = px.scatter(df, x='Investimento', y='Lucro',
                            title="Investimento vs Lucro")
            z = np.polyfit(df['Investimento'], df['Lucro'], 1)
            x_line = np.linspace(df['Investimento'].min(), df['Investimento'].max(), 50)
            fig.add_trace(go.Scatter(x=x_line, y=z[0]*x_line + z[1],
                                    mode='lines', line=dict(color='red', dash='dash'),
                                    name='Tendência'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Viés de Seleção")
            
            st.markdown("""
            **O problema:** Quem recebe o tratamento já é diferente de quem não recebe.
            
            **Exemplo: Treinamento (voluntário) → Produtividade**
            - Funcionários motivados se inscrevem no treinamento
            - Funcionários motivados também são mais produtivos
            - O "efeito" do treinamento inclui o efeito da motivação
            """)
            
            selection_intensity = st.slider("Intensidade da seleção", 0.0, 1.0, 0.5, 0.1,
                                            key="selection_slider")
            
            df = make_selection_bias_data(n=200, selection_intensity=selection_intensity)
            
            # Comparação ingênua
            media_tratado = df.loc[df['Treinamento'] == 1, 'Produtividade'].mean()
            media_controle = df.loc[df['Treinamento'] == 0, 'Produtividade'].mean()
            diff = media_tratado - media_controle
            
            st.metric("Diferença média (tratado - controle)", f"{diff:.1f}",
                     delta=f"Viés: {diff - 10:.1f}" if abs(diff - 10) > 1 else "≈ verdadeiro")
            st.caption("Efeito verdadeiro: 10")
        
        with col2:
            fig = px.box(df, x='Treinamento', y='Produtividade', color='Treinamento',
                        title="Produtividade por Status de Treinamento")
            fig.update_xaxes(tickvals=[0, 1], ticktext=['Sem Treinamento', 'Com Treinamento'])
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔬 Simulador de Ameaças")
        
        st.markdown("**Ative/desative cada ameaça e veja o viés resultante:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ameaca_omissao = st.checkbox("Variável Omitida", value=True)
        with col2:
            ameaca_reversa = st.checkbox("Causalidade Reversa", value=False)
        with col3:
            ameaca_selecao = st.checkbox("Viés de Seleção", value=False)
        
        # Calcular viés combinado (simplificado)
        vies_total = 0
        if ameaca_omissao:
            vies_total += 1.5
        if ameaca_reversa:
            vies_total += 2.0
        if ameaca_selecao:
            vies_total += 3.0
        
        efeito_verdadeiro = 5
        efeito_estimado = efeito_verdadeiro + vies_total
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Efeito Verdadeiro", f"{efeito_verdadeiro:.1f}")
        col2.metric("Efeito Estimado", f"{efeito_estimado:.1f}")
        col3.metric("Viés Total", f"+{vies_total:.1f}", delta_color="inverse")
        
        if vies_total > 0:
            st.error(f"""
            🚨 Com as ameaças ativas, você superestimaria o efeito em {(vies_total/efeito_verdadeiro)*100:.0f}%!
            Decisões baseadas nesse número seriam enganosas.
            """)
        else:
            st.success("✅ Sem ameaças, a estimativa seria não-viesada.")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Mapeia possíveis ameaças antes de confiar em qualquer estimativa
    - Pergunta: "O que mais poderia explicar esse resultado?"
    """)


def render_section_S5():
    """S5: Estratégias Práticas de Identificação"""
    st.header("🛠️ Estratégias Práticas de Identificação")
    
    st.markdown("""
    Como obter evidência causal no mundo real? Aqui estão as principais estratégias:
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Antes/Depois", "👥 Grupos de Controle", 
                                       "🎲 Teste A/B", "📐 Quase-Experimento"])
    
    with tab1:
        st.subheader("Comparação Antes vs Depois")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Ideia:** Comparar resultados antes e depois da intervenção.
            
            **Limitação principal:** Não controla tendências ou outros eventos simultâneos.
            
            **Quando funciona:**
            - Intervenção foi inesperada
            - Não há tendência pré-existente clara
            - Não houve outros eventos no período
            """)
            
            trend_before = st.slider("Tendência pré-existente", -2.0, 2.0, 0.0, 0.5,
                                     key="trend_slider")
        
        with col2:
            np.random.seed(42)
            n = 24
            t = np.arange(1, n+1)
            intervention = 12
            
            # Com tendência
            y = 50 + trend_before * t + np.random.normal(0, 3, n)
            y[intervention:] += 10  # Efeito verdadeiro
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t[:intervention], y=y[:intervention],
                                    mode='lines+markers', name='Antes', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=t[intervention:], y=y[intervention:],
                                    mode='lines+markers', name='Depois', line=dict(color='green')))
            fig.add_vline(x=intervention, line_dash="dash", annotation_text="Intervenção")
            
            # Mostrar tendência projetada
            y_projected = 50 + trend_before * t
            fig.add_trace(go.Scatter(x=t, y=y_projected, mode='lines',
                                    line=dict(color='red', dash='dot'),
                                    name='Tendência'))
            
            fig.update_layout(title="Antes vs Depois (com tendência)", height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            media_antes = np.mean(y[:intervention])
            media_depois = np.mean(y[intervention:])
            
            st.metric("Diferença Antes/Depois", f"{media_depois - media_antes:.1f}",
                     help="Pode confundir tendência com efeito!")
    
    with tab2:
        st.subheader("Grupos de Controle")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Ideia:** Comparar quem recebeu tratamento com quem não recebeu.
            
            **Limitação principal:** Grupos podem não ser comparáveis.
            
            **Quando funciona:**
            - Atribuição foi aleatória ou "como se" aleatória
            - Grupos são similares em características observáveis
            - Não há spillovers (tratamento de um não afeta o outro)
            """)
            
            st.markdown("""
            **Diferença-em-Diferenças (DiD):**
            
            Combina antes/depois COM grupos de controle:
            
            Efeito = (Tratado_depois - Tratado_antes) - (Controle_depois - Controle_antes)
            
            Remove tendências comuns aos dois grupos.
            """)
        
        with col2:
            np.random.seed(42)
            n_t = 12
            t = np.arange(1, n_t*2 + 1)
            
            # Grupo tratado
            y_tratado = 50 + 0.5 * t + np.random.normal(0, 2, n_t*2)
            y_tratado[n_t:] += 15
            
            # Grupo controle (mesma tendência, sem tratamento)
            y_controle = 45 + 0.5 * t + np.random.normal(0, 2, n_t*2)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y_tratado, mode='lines+markers',
                                    name='Tratado', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=t, y=y_controle, mode='lines+markers',
                                    name='Controle', line=dict(color='orange')))
            fig.add_vline(x=n_t, line_dash="dash", annotation_text="Tratamento")
            
            fig.update_layout(title="Tratado vs Controle", height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # DiD
            diff_tratado = np.mean(y_tratado[n_t:]) - np.mean(y_tratado[:n_t])
            diff_controle = np.mean(y_controle[n_t:]) - np.mean(y_controle[:n_t])
            did = diff_tratado - diff_controle
            
            st.metric("Estimativa DiD", f"{did:.1f}", help="Efeito verdadeiro: 15")
    
    with tab3:
        st.subheader("🎲 Teste A/B (Experimento Randomizado)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **O padrão ouro:** Atribuição aleatória ao tratamento.
            
            **Por que funciona:**
            - Aleatorização garante que grupos são comparáveis em média
            - Elimina viés de seleção
            - Permite estimativa não-viesada do efeito causal
            
            **Na prática:**
            - Metade dos usuários vê versão A (controle)
            - Metade vê versão B (tratamento)
            - Compara taxas de conversão
            """)
            
            true_effect = st.slider("Efeito verdadeiro (%)", 0, 10, 5, 1, key="ab_effect")
            n_total = st.slider("Tamanho da amostra", 500, 5000, 1000, 500, key="ab_n")
        
        with col2:
            df = make_ab_test_sim(n_total=n_total, true_effect=true_effect)
            
            # Resultados
            conv_tratamento = df.loc[df['Tratamento'] == 1, 'Conversao'].mean() * 100
            conv_controle = df.loc[df['Tratamento'] == 0, 'Conversao'].mean() * 100
            diff = conv_tratamento - conv_controle
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Conversão Controle", f"{conv_controle:.1f}%")
            col_m2.metric("Conversão Tratamento", f"{conv_tratamento:.1f}%")
            col_m3.metric("Efeito Estimado", f"{diff:.2f}%",
                         delta=f"vs verdadeiro: {true_effect}%")
            
            # Gráfico de barras
            fig = px.bar(x=['Controle', 'Tratamento'], y=[conv_controle, conv_tratamento],
                        title="Taxa de Conversão por Grupo",
                        labels={'x': 'Grupo', 'y': 'Conversão (%)'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📐 Quase-Experimento (Regression Discontinuity)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Ideia:** Explorar regras de corte que criam tratamento "quase aleatório".
            
            **Exemplo:**
            - Bolsa de estudos para nota ≥ 70
            - Crédito para score ≥ 650
            - Promoção para vendas ≥ meta
            
            **Lógica:** Pessoas logo acima e logo abaixo do corte são muito similares,
            exceto pelo tratamento. É como um experimento natural.
            """)
            
            cutoff = st.slider("Ponto de corte", 30, 70, 50, 5, key="rd_cutoff")
            effect = st.slider("Efeito do tratamento", 0, 30, 15, 5, key="rd_effect")
        
        with col2:
            df = make_quasi_experiment_data(n=300, cutoff=cutoff, effect=effect)
            
            fig = px.scatter(df, x='Score', y='Resultado', color='Grupo',
                            title=f"Regression Discontinuity (corte em {cutoff})")
            fig.add_vline(x=cutoff, line_dash="dash", line_color="red")
            
            # Linhas de tendência separadas
            df_below = df[df['Score'] < cutoff]
            df_above = df[df['Score'] >= cutoff]
            
            if len(df_below) > 2:
                z1 = np.polyfit(df_below['Score'], df_below['Resultado'], 1)
                x1 = np.linspace(df_below['Score'].min(), cutoff, 50)
                fig.add_trace(go.Scatter(x=x1, y=z1[0]*x1 + z1[1], mode='lines',
                                        line=dict(color='blue'), showlegend=False))
            
            if len(df_above) > 2:
                z2 = np.polyfit(df_above['Score'], df_above['Resultado'], 1)
                x2 = np.linspace(cutoff, df_above['Score'].max(), 50)
                fig.add_trace(go.Scatter(x=x2, y=z2[0]*x2 + z2[1], mode='lines',
                                        line=dict(color='green'), showlegend=False))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Estimar efeito no ponto de corte
            window = 5
            close_below = df[(df['Score'] >= cutoff - window) & (df['Score'] < cutoff)]
            close_above = df[(df['Score'] >= cutoff) & (df['Score'] < cutoff + window)]
            
            if len(close_below) > 0 and len(close_above) > 0:
                rd_effect = close_above['Resultado'].mean() - close_below['Resultado'].mean()
                st.metric("Efeito estimado (RD)", f"{rd_effect:.1f}",
                         help=f"Verdadeiro: {effect}")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Planeja experimentos randomizados sempre que possível
    - Busca "experimentos naturais" quando randomização não é viável
    """)


def render_section_S6():
    """S6: Aplicações em Negócios"""
    st.header("💼 Aplicações em Negócios")
    
    tab1, tab2, tab3 = st.tabs(["📣 Marketing", "🏦 Finanças", "👥 RH"])
    
    with tab1:
        st.subheader("📣 Marketing: Impacto de Campanhas")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Perguntas causais típicas:**
            - A campanha aumentou vendas?
            - O desconto reduziu churn?
            - O email marketing gera conversões?
            
            **Ameaças comuns:**
            - Sazonalidade confunde antes/depois
            - Clientes que recebem oferta já eram mais engajados
            - Outros eventos simultâneos (concorrente, economia)
            
            **Estratégias recomendadas:**
            - **Teste A/B:** Sorteio de quem recebe a campanha
            - **Holdout:** Reservar grupo sem tratamento
            - **Geo-experimentos:** Testar em algumas cidades
            """)
        
        with col2:
            st.markdown("""
            **Exemplo: Campanha de Email**
            
            | Abordagem | O que compara | Problema |
            |-----------|---------------|----------|
            | Antes/depois | Vendas mês passado vs este | Sazonalidade |
            | Quem abriu vs não abriu | Clientes que clicaram | Seleção |
            | **Teste A/B** | Sorteados vs não sorteados | ✅ Válido |
            
            **O que muda na decisão:**
            - Com identificação: "A campanha gera R$ X por cliente"
            - Sem identificação: "Clientes da campanha compraram mais" (pode ser ilusão)
            """)
            
            st.success("""
            ✅ **Recomendação:** Sempre reserve um grupo de controle 
            (mesmo que pequeno) para validar o impacto real.
            """)
    
    with tab2:
        st.subheader("🏦 Finanças: Crédito e Risco")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Perguntas causais típicas:**
            - Aumentar limite reduz inadimplência?
            - Juros mais baixos aumentam demanda?
            - Alerta de fraude previne perdas?
            
            **Ameaças comuns:**
            - Clientes que recebem mais limite são os melhores
            - Quem busca crédito em crise é diferente
            - Políticas mudam com ambiente econômico
            
            **Estratégias recomendadas:**
            - **RDD:** Explorar cutoffs de score
            - **Variação de política:** Mudanças exógenas de regulação
            - **Experimentos:** Quando eticamente possível
            """)
        
        with col2:
            st.markdown("""
            **Exemplo: Efeito do Limite de Crédito**
            
            | Abordagem | Resultado |
            |-----------|-----------|
            | Correlação limite × inadimplência | "Mais limite = menos default" |
            | Problema | Melhores clientes ganham mais limite! |
            | Solução (RDD) | Comparar clientes no limiar do score |
            
            **O que muda na decisão:**
            - Com identificação: "Aumentar limite em 10% reduz default em X%"
            - Sem identificação: Você pode estar dando mais limite para quem já não daria default
            """)
            
            st.warning("""
            ⚠️ **Cuidado:** Em finanças, viés de seleção é a regra.
            Bancos dão crédito para quem tem menor risco.
            """)
    
    with tab3:
        st.subheader("👥 RH: Treinamento e Produtividade")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Perguntas causais típicas:**
            - Treinamento aumenta produtividade?
            - Promoção aumenta engajamento?
            - Home office afeta performance?
            
            **Ameaças comuns:**
            - Funcionários motivados buscam treinamento
            - Promovidos já eram os melhores
            - Quem vai para home office é diferente
            
            **Estratégias recomendadas:**
            - **Sorteio de vagas:** Quando demanda > oferta
            - **Matching:** Comparar similares tratados vs não
            - **DiD:** Comparar mudança antes/depois entre grupos
            """)
        
        with col2:
            st.markdown("""
            **Exemplo: Programa de Treinamento**
            
            | Métrica | Sem Identificação | Com Identificação |
            |---------|-------------------|-------------------|
            | Produtividade pós-treino | +20% | +8% |
            | Diferença | Inclui efeito de seleção | Efeito causal real |
            | Decisão | ROI superestimado | ROI realista |
            
            **O que muda na decisão:**
            - Com identificação: Saber se vale investir em mais treinamento
            - Sem identificação: Pode estar jogando dinheiro fora
            """)
            
            st.info("""
            💡 **Dica prática:** Se a demanda por treinamento excede as vagas,
            use loteria para selecionar. Isso cria um experimento natural!
            """)
    
    st.markdown("---")
    
    st.subheader("📋 Resumo: O que Muda na Decisão?")
    
    resumo_df = pd.DataFrame({
        'Área': ['Marketing', 'Finanças', 'RH'],
        'Pergunta Típica': ['Campanha aumenta vendas?', 'Limite reduz default?', 'Treino aumenta produtividade?'],
        'Ameaça Principal': ['Sazonalidade, seleção', 'Viés de concessão', 'Auto-seleção'],
        'Estratégia Recomendada': ['Teste A/B', 'RDD em score', 'Sorteio de vagas']
    })
    st.dataframe(resumo_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Exige evidência causal antes de escalar programas
    - Planeja avaliações de impacto desde o início de iniciativas
    """)


def render_section_S7():
    """S7: Resumo Executivo e Ponte para o Próximo Módulo"""
    st.header("📋 Resumo Executivo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### O que Aprendemos sobre Causalidade
        
        ✅ **Correlação ≠ Causalidade:**
        - Associação estatística não prova que X causa Y
        - Confundidores podem criar correlações espúrias
        - Decisões baseadas só em correlação podem falhar
        
        ✅ **O Problema do Contrafactual:**
        - Para medir efeito, precisamos saber o que teria acontecido sem X
        - Nunca observamos o contrafactual diretamente
        - Estratégias de identificação aproximam o contrafactual
        
        ✅ **Principais Ameaças:**
        - **Variável omitida:** Fator oculto causa X e Y
        - **Causalidade reversa:** Y causa X, não o contrário
        - **Viés de seleção:** Tratados são diferentes dos controles
        
        ✅ **Estratégias de Identificação:**
        - **Teste A/B:** Randomização garante comparabilidade
        - **DiD:** Antes/depois + controle remove tendências
        - **RDD:** Explorar cutoffs para efeito local
        - **Quase-experimentos:** Choques exógenos e regras
        
        ✅ **Implicações para Decisão:**
        - Padrões não bastam — precisamos de efeitos causais
        - Investimento sem identificação pode ser desperdício
        - Planejar avaliação ANTES de implementar
        """)
    
    with col2:
        st.markdown("### 💡 Mensagem Final")
        
        st.info("""
        **"Padrões explicam pouco. Causalidade orienta decisões."**
        
        Ver correlação é fácil.
        Provar causalidade é difícil.
        Decidir sem causalidade é arriscado.
        """)
        
        st.markdown("### 🧪 Quiz Final")
        
        st.markdown("""
        Uma rede de varejo observou que lojas com 
        gerentes que fizeram MBA têm faturamento 15% maior.
        """)
        
        resposta = st.radio(
            "O que você recomendaria?",
            ["Enviar todos gerentes para fazer MBA",
             "Investigar se há viés de seleção ou confundidores",
             "Correlação prova que MBA causa sucesso"],
            key="quiz_final"
        )
        
        if st.button("Ver análise", key="btn_final"):
            if resposta == "Investigar se há viés de seleção ou confundidores":
                st.success("""
                ✅ **Correto!** Possíveis problemas:
                - **Seleção:** Gerentes melhores buscam MBA
                - **Confundidor:** Lojas maiores têm gerentes com MBA
                - **Causalidade reversa:** Lojas lucrativas pagam MBA
                
                Antes de investir, faça um piloto randomizado!
                """)
            else:
                st.error("Cuidado! Correlação não prova que MBA causa o sucesso.")
    
    st.markdown("---")
    
    st.subheader("🔜 Próximo Módulo: Séries Temporais")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Modelos ARIMA:**
        - Autoregressivo
        - Integrado
        - Média Móvel
        """)
    
    with col2:
        st.markdown("""
        **Estacionariedade:**
        - Raiz unitária
        - Tendências
        - Sazonalidade
        """)
    
    with col3:
        st.markdown("""
        **Forecasting:**
        - Previsão pontual
        - Intervalos
        - Avaliação
        """)
    
    st.success("""
    🎓 **Conclusão:** Antes de agir com base em dados, pergunte-se:
    "Isso é uma correlação ou um efeito causal? Qual a minha estratégia de identificação?"
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Inclui "identificação causal" como critério em análises
    - Exige grupo de controle ou experimento antes de escalar iniciativas
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("🔍 Módulo 5: Causalidade e Identificação")
    st.markdown("**Laboratório de Econometria** | De Correlação a Decisão Causal")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Mostrar por que decisões estratégicas são hipóteses causais
        - Distinguir **correlação** de **causalidade** em negócios
        - Introduzir o conceito de **contrafactual**
        - Explicar **identificação** e principais ameaças
        - Apresentar estratégias práticas: **A/B, DiD, RDD**
        - Conectar causalidade a aplicações em marketing, finanças e RH
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "🔗 Correlação ≠ Causalidade",
        "S2": "🔮 O Contrafactual",
        "S3": "🎯 Identificação",
        "S4": "⚠️ Ameaças à Causalidade",
        "S5": "🛠️ Estratégias Práticas",
        "S6": "💼 Aplicações em Negócios",
        "S7": "📋 Resumo e Próximos Passos"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Dica:** Causalidade é fundamental 
    para decisões baseadas em evidência.
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
            page_title="Módulo 5: Causalidade e Identificação",
            page_icon="🔍",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()