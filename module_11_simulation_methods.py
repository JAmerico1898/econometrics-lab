"""
Laboratório de Econometria - Module 11: Simulation Methods
Aplicativo educacional interativo para Monte Carlo, Bootstrap e técnicas de simulação.
Público-alvo: alunos de MBA com perfis quantitativos heterogêneos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import time

# =============================================================================
# FUNÇÕES AUXILIARES PARA SIMULAÇÃO E CÁLCULOS
# =============================================================================

def mc_estimate_stat(distribution: str = 'normal', n_samples: int = 1000,
                     stat: str = 'mean', params: dict = None, seed: int = None) -> dict:
    """
    Estima uma estatística via Monte Carlo.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if params is None:
        params = {}
    
    # Gerar amostras
    if distribution == 'normal':
        samples = np.random.normal(params.get('mu', 0), params.get('sigma', 1), n_samples)
    elif distribution == 't':
        samples = np.random.standard_t(params.get('df', 5), n_samples)
    elif distribution == 'lognormal':
        samples = np.random.lognormal(params.get('mu', 0), params.get('sigma', 0.5), n_samples)
    else:
        samples = np.random.normal(0, 1, n_samples)
    
    # Calcular estatística
    if stat == 'mean':
        estimate = np.mean(samples)
        se = np.std(samples) / np.sqrt(n_samples)
    elif stat == 'median':
        estimate = np.median(samples)
        se = 1.253 * np.std(samples) / np.sqrt(n_samples)  # Aproximação
    elif stat == 'percentile_5':
        estimate = np.percentile(samples, 5)
        se = np.std(samples) / np.sqrt(n_samples) * 2  # Aproximação
    elif stat == 'percentile_95':
        estimate = np.percentile(samples, 95)
        se = np.std(samples) / np.sqrt(n_samples) * 2
    else:
        estimate = np.mean(samples)
        se = np.std(samples) / np.sqrt(n_samples)
    
    return {
        'estimate': estimate,
        'se': se,
        'samples': samples,
        'n': n_samples
    }


def standard_error_mc(variance: float, n: int) -> float:
    """Calcula erro padrão do Monte Carlo: sqrt(var/n)."""
    return np.sqrt(variance / n)


def simulate_gbm_paths(S0: float, mu: float, sigma: float, T: float,
                       n_steps: int, n_paths: int, seed: int = None) -> np.ndarray:
    """
    Simula trajetórias de preço via Geometric Brownian Motion.
    dS = mu*S*dt + sigma*S*dW
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0
    
    for t in range(1, n_steps + 1):
        z = np.random.normal(0, 1, n_paths)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return paths


def price_option_mc(S0: float, K: float, r: float, sigma: float, T: float,
                    n_paths: int, option_type: str = 'call', seed: int = None) -> dict:
    """
    Precifica opção europeia via Monte Carlo.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Simular preços finais
    z = np.random.normal(0, 1, n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    
    # Payoff
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Preço descontado
    price = np.exp(-r * T) * np.mean(payoffs)
    se = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return {
        'price': price,
        'se': se,
        'ST': ST,
        'payoffs': payoffs,
        'n_paths': n_paths
    }


def black_scholes_price(S0: float, K: float, r: float, sigma: float, T: float,
                        option_type: str = 'call') -> float:
    """Fórmula de Black-Scholes para referência."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)
    
    return price


def simulate_fat_tails(n: int = 1000, distribution: str = 'normal', 
                       df: int = 4, seed: int = None) -> np.ndarray:
    """Simula retornos com caudas normais ou pesadas."""
    if seed is not None:
        np.random.seed(seed)
    
    if distribution == 'normal':
        returns = np.random.normal(0, 0.02, n)
    elif distribution == 't':
        returns = np.random.standard_t(df, n) * 0.02 / np.sqrt(df / (df - 2))
    else:
        returns = np.random.normal(0, 0.02, n)
    
    return returns


def compute_var_es(returns: np.ndarray, confidence: float = 0.95) -> dict:
    """Calcula VaR e Expected Shortfall."""
    alpha = 1 - confidence
    var = -np.percentile(returns, alpha * 100)
    
    # ES: média das perdas além do VaR
    losses = -returns
    es = np.mean(losses[losses >= var])
    
    return {
        'var': var,
        'es': es,
        'alpha': alpha
    }


def antithetic_variates_mc(S0: float, K: float, r: float, sigma: float, T: float,
                           n_paths: int, option_type: str = 'call', seed: int = None) -> dict:
    """
    Monte Carlo com variáveis antitéticas.
    Usa z e -z para reduzir variância.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_pairs = n_paths // 2
    z = np.random.normal(0, 1, n_pairs)
    
    # Preços com z e -z
    ST_pos = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    ST_neg = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (-z))
    
    # Payoffs
    if option_type == 'call':
        payoffs_pos = np.maximum(ST_pos - K, 0)
        payoffs_neg = np.maximum(ST_neg - K, 0)
    else:
        payoffs_pos = np.maximum(K - ST_pos, 0)
        payoffs_neg = np.maximum(K - ST_neg, 0)
    
    # Média dos pares
    payoffs_avg = (payoffs_pos + payoffs_neg) / 2
    
    price = np.exp(-r * T) * np.mean(payoffs_avg)
    se = np.exp(-r * T) * np.std(payoffs_avg) / np.sqrt(n_pairs)
    
    return {
        'price': price,
        'se': se,
        'n_paths': n_paths
    }


def control_variate_mc(S0: float, K: float, r: float, sigma: float, T: float,
                       n_paths: int, option_type: str = 'call', seed: int = None) -> dict:
    """
    Monte Carlo com variável de controle.
    Usa preço do ativo como controle (valor esperado conhecido: S0*e^rT).
    """
    if seed is not None:
        np.random.seed(seed)
    
    z = np.random.normal(0, 1, n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    
    # Payoff
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Controle: ST descontado
    control = ST * np.exp(-r * T)
    expected_control = S0  # E[ST * e^-rT] = S0
    
    # Coeficiente ótimo
    cov_pc = np.cov(payoffs, control)[0, 1]
    var_c = np.var(control)
    beta = cov_pc / var_c if var_c > 0 else 0
    
    # Estimador ajustado
    payoffs_adj = payoffs - beta * (control - expected_control)
    
    price = np.exp(-r * T) * np.mean(payoffs_adj)
    se = np.exp(-r * T) * np.std(payoffs_adj) / np.sqrt(n_paths)
    
    return {
        'price': price,
        'se': se,
        'beta': beta,
        'n_paths': n_paths
    }


def quasi_mc_low_discrepancy(S0: float, K: float, r: float, sigma: float, T: float,
                              n_paths: int, option_type: str = 'call') -> dict:
    """
    Quasi-Monte Carlo com sequência de baixa discrepância (Halton simplificado).
    """
    # Sequência de Halton base 2 (simplificada)
    def halton_sequence(n, base=2):
        seq = np.zeros(n)
        for i in range(n):
            f = 1
            r = 0
            index = i + 1
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index = index // base
            seq[i] = r
        return seq
    
    # Gerar sequência e converter para Normal
    u = halton_sequence(n_paths)
    z = stats.norm.ppf(np.clip(u, 0.001, 0.999))
    
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    se = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return {
        'price': price,
        'se': se,
        'n_paths': n_paths
    }


def bootstrap_resample(data: np.ndarray, n_bootstrap: int = 1000, 
                       stat_func: callable = np.mean, seed: int = None) -> np.ndarray:
    """
    Bootstrap: reamostra dados com reposição e calcula estatística.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[b] = stat_func(resample)
    
    return bootstrap_stats


def bootstrap_ci(bootstrap_stats: np.ndarray, confidence: float = 0.95) -> dict:
    """Calcula intervalo de confiança percentílico do bootstrap."""
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return {
        'lower': lower,
        'upper': upper,
        'mean': np.mean(bootstrap_stats),
        'se': np.std(bootstrap_stats)
    }


def var_bootstrap(returns: np.ndarray, confidence: float = 0.95, 
                  n_bootstrap: int = 1000, seed: int = None) -> dict:
    """Calcula VaR via bootstrap."""
    if seed is not None:
        np.random.seed(seed)
    
    alpha = 1 - confidence
    
    def var_stat(data):
        return -np.percentile(data, alpha * 100)
    
    bootstrap_vars = bootstrap_resample(returns, n_bootstrap, var_stat, seed)
    ci = bootstrap_ci(bootstrap_vars, 0.95)
    
    return {
        'var_mean': ci['mean'],
        'var_lower': ci['lower'],
        'var_upper': ci['upper'],
        'var_se': ci['se'],
        'bootstrap_vars': bootstrap_vars
    }


def case_portfolio_sim(weights: np.ndarray, returns_matrix: np.ndarray,
                       n_sim: int = 10000, seed: int = None) -> dict:
    """
    Simula retornos de portfólio para análise de risco.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Estatísticas dos ativos
    mean_returns = np.mean(returns_matrix, axis=0)
    cov_matrix = np.cov(returns_matrix.T)
    
    # Simular retornos do portfólio
    simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_sim)
    portfolio_returns = simulated_returns @ weights
    
    # Métricas
    var_95 = -np.percentile(portfolio_returns, 5)
    es_95 = -np.mean(portfolio_returns[portfolio_returns < -var_95])
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
    
    return {
        'portfolio_returns': portfolio_returns,
        'var_95': var_95,
        'es_95': es_95,
        'sharpe': sharpe,
        'mean': np.mean(portfolio_returns),
        'std': np.std(portfolio_returns)
    }


def log_params_and_seed(params: dict, seed: int) -> str:
    """Gera log de parâmetros para reprodutibilidade."""
    log = f"=== Log de Simulação ===\n"
    log += f"Seed: {seed}\n"
    log += f"Parâmetros:\n"
    for key, value in params.items():
        log += f"  {key}: {value}\n"
    log += f"========================"
    return log


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO POR SEÇÃO
# =============================================================================

def render_section_S1():
    """S1: A Lógica da Simulação para Tomada de Decisão"""
    st.header("🎲 A Lógica da Simulação")
    
    st.markdown("""
    **Simulação** é uma ferramenta de decisão quando:
    - Fórmulas fechadas não existem ou são muito complexas
    - Premissas de modelos tradicionais são frágeis
    - Queremos entender a **distribuição completa** de resultados
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Analogia Gerencial")
        
        st.markdown("""
        **Simulação = "Test-Drive" antes de lançar**
        
        Antes de:
        - Lançar um produto → Simula demanda
        - Implementar estratégia → Simula cenários
        - Aprovar crédito → Simula inadimplência
        - Precificar derivativo → Simula trajetórias
        
        **Vantagens:**
        - Ambiente controlado
        - Testa múltiplos cenários
        - Quantifica incerteza
        - Sem risco real
        """)
        
        st.subheader("Dados Reais vs Simulação")
        
        st.markdown("""
        | Aspecto | Dados Reais | Simulação |
        |---------|-------------|-----------|
        | Ruído | Alto | Controlado |
        | Tamanho | Limitado | Ilimitado |
        | Cenários extremos | Raros | Geráveis |
        | Custo de erro | Alto | Zero |
        """)
    
    with col2:
        st.subheader("Mini-Exemplo: Lucro com Demanda Incerta")
        
        st.markdown("""
        **Cenário:** Lançar produto com custo fixo de R$ 100k
        - Preço: R$ 50/unidade
        - Custo variável: R$ 30/unidade
        - Demanda incerta: Normal(5000, 1500)
        """)
        
        n_sim = st.slider("Número de simulações", 100, 10000, 1000, key="n_sim_intro")
        
        np.random.seed(42)
        demanda = np.random.normal(5000, 1500, n_sim)
        demanda = np.maximum(demanda, 0)  # Não pode ser negativa
        
        lucro = (50 - 30) * demanda - 100000
        
        fig = px.histogram(lucro / 1000, nbins=50, 
                          labels={'value': 'Lucro (R$ mil)', 'count': 'Frequência'},
                          title="Distribuição do Lucro Simulado")
        fig.add_vline(x=0, line_dash="dash", line_color="red",
                     annotation_text="Break-even")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        prob_prejuizo = np.mean(lucro < 0) * 100
        lucro_medio = np.mean(lucro) / 1000
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Lucro médio", f"R$ {lucro_medio:.0f}k")
        col_m2.metric("P(Prejuízo)", f"{prob_prejuizo:.1f}%")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa simulação antes de decisões com alta incerteza
    - Quantifica não só o valor esperado, mas a distribuição de resultados
    - Toma decisões informadas sobre risco vs retorno
    """)


def render_section_S2():
    """S2: Monte Carlo: Amostragem Aleatória e Convergência"""
    st.header("🎰 Monte Carlo: Amostragem e Convergência")
    
    st.markdown("""
    **Monte Carlo:** Repetir sorteios de uma distribuição para estimar um resultado.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuração")
        
        distribution = st.selectbox(
            "Distribuição:",
            ['normal', 't', 'lognormal'],
            key="dist_mc"
        )
        
        stat = st.selectbox(
            "Estatística a estimar:",
            ['mean', 'median', 'percentile_5', 'percentile_95'],
            format_func=lambda x: {'mean': 'Média', 'median': 'Mediana',
                                  'percentile_5': 'Percentil 5%', 
                                  'percentile_95': 'Percentil 95%'}[x],
            key="stat_mc"
        )
        
        n_samples = st.slider("Número de amostras (N)", 100, 50000, 1000, 100, key="n_mc")
        
        seed = st.number_input("Seed (para reprodutibilidade)", 1, 9999, 42, key="seed_mc")
        
        st.markdown("""
        **Erro Padrão do Monte Carlo:**
        $$SE = \\sqrt{\\frac{Var(X)}{N}}$$
        
        **Regra prática:**
        > "10x menos erro exige 100x mais simulações"
        """)
    
    with col2:
        # Simular
        result = mc_estimate_stat(distribution, n_samples, stat, 
                                 {'mu': 0, 'sigma': 1, 'df': 5}, seed)
        
        st.subheader("Resultados")
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Estimativa", f"{result['estimate']:.4f}")
        col_m2.metric("Erro Padrão", f"{result['se']:.4f}")
        
        # Histograma das amostras
        fig = px.histogram(result['samples'], nbins=50,
                          labels={'value': 'Valor', 'count': 'Frequência'},
                          title=f"Distribuição das Amostras (N={n_samples})")
        fig.add_vline(x=result['estimate'], line_dash="dash", line_color="red",
                     annotation_text=f"Estimativa: {result['estimate']:.3f}")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Convergência
    st.subheader("Convergência com N")
    
    ns = [100, 500, 1000, 2000, 5000, 10000, 20000]
    estimates = []
    ses = []
    
    for n in ns:
        r = mc_estimate_stat(distribution, n, stat, {'mu': 0, 'sigma': 1, 'df': 5}, seed)
        estimates.append(r['estimate'])
        ses.append(r['se'])
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Estimativa vs N", "Erro Padrão vs N"])
    
    fig.add_trace(go.Scatter(x=ns, y=estimates, mode='lines+markers', name='Estimativa'),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=ns, y=ses, mode='lines+markers', name='SE'),
                 row=1, col=2)
    
    fig.update_xaxes(type='log', title_text='N (log)', row=1, col=1)
    fig.update_xaxes(type='log', title_text='N (log)', row=1, col=2)
    fig.update_yaxes(type='log', title_text='SE (log)', row=1, col=2)
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"""
    💡 **Observe:** O erro padrão cai como 1/√N. 
    Para reduzir o erro de {ses[0]:.4f} para {ses[-1]:.4f}, 
    precisamos aumentar N de {ns[0]} para {ns[-1]} ({ns[-1]//ns[0]}x mais simulações).
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Escolhe N balanceando precisão e custo computacional
    - Sabe que para alta precisão, precisa de muitas simulações
    - Reporta resultados com intervalo de confiança (não só ponto)
    """)


def render_section_S3():
    """S3: Monte Carlo na Prática Financeira: Precificação de Opções"""
    st.header("📈 Monte Carlo: Precificação de Opções")
    
    st.markdown("""
    Uma aplicação clássica: precificar opções simulando trajetórias de preço.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Parâmetros da Opção")
        
        S0 = st.slider("Preço inicial (S₀)", 50.0, 150.0, 100.0, 5.0, key="S0_opt")
        K = st.slider("Strike (K)", 50.0, 150.0, 100.0, 5.0, key="K_opt")
        r = st.slider("Taxa livre de risco (r)", 0.01, 0.15, 0.05, 0.01, key="r_opt")
        sigma = st.slider("Volatilidade (σ)", 0.1, 0.5, 0.2, 0.05, key="sigma_opt")
        T = st.slider("Tempo até vencimento (anos)", 0.25, 2.0, 1.0, 0.25, key="T_opt")
        
        option_type = st.radio("Tipo de opção:", ['call', 'put'], horizontal=True, key="type_opt")
        
        n_paths = st.slider("Número de trajetórias", 1000, 100000, 10000, 1000, key="n_paths_opt")
        
        seed_opt = st.number_input("Seed", 1, 9999, 42, key="seed_opt")
    
    with col2:
        # Simular trajetórias para visualização
        n_steps = 50
        n_visual = min(100, n_paths)
        
        paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_visual, seed_opt)
        
        fig = go.Figure()
        for i in range(n_visual):
            fig.add_trace(go.Scatter(y=paths[:, i], mode='lines',
                                    line=dict(width=0.5), showlegend=False,
                                    opacity=0.3))
        fig.add_hline(y=K, line_dash="dash", line_color="red",
                     annotation_text=f"Strike K={K}")
        fig.update_layout(
            title=f"Trajetórias de Preço (GBM, {n_visual} de {n_paths})",
            xaxis_title="Passos de tempo",
            yaxis_title="Preço",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Precificar
    start_time = time.time()
    mc_result = price_option_mc(S0, K, r, sigma, T, n_paths, option_type, seed_opt)
    mc_time = time.time() - start_time
    
    bs_price = black_scholes_price(S0, K, r, sigma, T, option_type)
    
    st.subheader("Resultado da Precificação")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Preço MC", f"${mc_result['price']:.4f}")
    col2.metric("Erro Padrão", f"${mc_result['se']:.4f}")
    col3.metric("Black-Scholes", f"${bs_price:.4f}")
    col4.metric("Diferença", f"${mc_result['price'] - bs_price:.4f}")
    
    # Distribuição de payoffs
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = px.histogram(mc_result['payoffs'], nbins=50,
                          labels={'value': 'Payoff', 'count': 'Frequência'},
                          title="Distribuição dos Payoffs")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(mc_result['ST'], nbins=50,
                          labels={'value': 'S_T', 'count': 'Frequência'},
                          title="Distribuição do Preço Final")
        fig.add_vline(x=K, line_dash="dash", line_color="red",
                     annotation_text="Strike")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Tempo de execução: {mc_time:.3f}s")
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa MC para opções exóticas onde BS não se aplica
    - Verifica convergência comparando com soluções conhecidas
    - Aumenta N quando precisa de mais precisão
    """)


def render_section_S4():
    """S4: Caudas Longas (Fat Tails) e Realismo de Mercado"""
    st.header("📊 Caudas Longas e Realismo")
    
    st.markdown("""
    A distribuição Normal subestima eventos extremos. 
    Mercados têm **caudas pesadas** que afetam VaR e precificação.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Normal vs Caudas Pesadas")
        
        st.markdown("""
        **Black-Scholes assume:**
        - Retornos log-normais
        - Volatilidade constante
        
        **Realidade:**
        - Caudas pesadas (curtose > 3)
        - Volatility clustering
        - Crashes mais frequentes
        """)
        
        distribution = st.radio(
            "Distribuição dos retornos:",
            ['normal', 't'],
            format_func=lambda x: 'Normal' if x == 'normal' else 't-Student (caudas pesadas)',
            key="dist_fat"
        )
        
        if distribution == 't':
            df = st.slider("Graus de liberdade (menor = caudas mais pesadas)", 
                          3, 30, 5, key="df_fat")
        else:
            df = 30
        
        n_returns = st.slider("Número de retornos", 1000, 50000, 10000, key="n_fat")
    
    with col2:
        # Simular
        returns = simulate_fat_tails(n_returns, distribution, df, seed=42)
        
        # VaR e ES
        metrics = compute_var_es(returns, 0.95)
        
        st.metric("VaR 95%", f"{metrics['var']*100:.2f}%")
        st.metric("Expected Shortfall 95%", f"{metrics['es']*100:.2f}%")
        
        curtose = stats.kurtosis(returns) + 3
        st.metric("Curtose", f"{curtose:.2f}", help="Normal = 3")
    
    # Comparação visual
    st.subheader("Comparação: Normal vs t-Student")
    
    np.random.seed(42)
    returns_normal = simulate_fat_tails(10000, 'normal', seed=42)
    returns_t = simulate_fat_tails(10000, 't', df=4, seed=42)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Histograma", "QQ-Plot vs Normal"])
    
    fig.add_trace(go.Histogram(x=returns_normal, name='Normal', opacity=0.5,
                              nbinsx=50), row=1, col=1)
    fig.add_trace(go.Histogram(x=returns_t, name='t(4)', opacity=0.5,
                              nbinsx=50), row=1, col=1)
    
    # QQ plot
    sorted_t = np.sort(returns_t)
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_t)))
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_t, mode='markers',
                            marker=dict(size=2), name='t(4)'), row=1, col=2)
    min_val, max_val = theoretical.min(), theoretical.max()
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val * 0.02, max_val * 0.02],
                            mode='lines', name='Normal', line=dict(dash='dash')),
                 row=1, col=2)
    
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    # Impacto no VaR
    st.subheader("Impacto no VaR: Normal vs Fat Tails")
    
    var_normal = compute_var_es(returns_normal, 0.95)
    var_t = compute_var_es(returns_t, 0.95)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Retornos Normais:**")
        st.metric("VaR 95%", f"{var_normal['var']*100:.2f}%")
        st.metric("ES 95%", f"{var_normal['es']*100:.2f}%")
    
    with col2:
        st.markdown("**Retornos t(4):**")
        st.metric("VaR 95%", f"{var_t['var']*100:.2f}%",
                 delta=f"+{(var_t['var']/var_normal['var']-1)*100:.0f}%")
        st.metric("ES 95%", f"{var_t['es']*100:.2f}%",
                 delta=f"+{(var_t['es']/var_normal['es']-1)*100:.0f}%")
    
    st.warning("""
    ⚠️ **Alerta:** Usar Normal quando caudas são pesadas **subestima o risco**.
    Capital regulatório e limites de risco podem estar inadequados!
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Verifica curtose dos retornos históricos
    - Usa distribuições com caudas pesadas para VaR/ES
    - Considera GARCH para volatilidade variável
    """)


def render_section_S5():
    """S5: Ganhando Eficiência: Técnicas de Redução de Variância"""
    st.header("⚡ Técnicas de Redução de Variância")
    
    st.markdown("""
    Podemos obter a **mesma precisão com menos simulações** usando técnicas inteligentes.
    """)
    
    tab1, tab2, tab3 = st.tabs(["🔄 Antitéticas", "🎯 Controle", "📐 Quasi-MC"])
    
    # Parâmetros comuns
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    n_paths = 10000
    seed = 42
    
    bs_price = black_scholes_price(S0, K, r, sigma, T, 'call')
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Variáveis Antitéticas")
            
            st.markdown("""
            **Ideia:** Para cada sorteio z, usar também -z.
            
            Os pares (z, -z) são **negativamente correlacionados**, 
            o que **reduz a variância** da média.
            
            **Como funciona:**
            - Sorteia z → calcula payoff₁
            - Usa -z → calcula payoff₂
            - Média: (payoff₁ + payoff₂) / 2
            
            **Ganho:** ~50% de redução de variância para muitos casos.
            """)
        
        with col2:
            # Comparar MC padrão vs antitético
            mc_std = price_option_mc(S0, K, r, sigma, T, n_paths, 'call', seed)
            mc_anti = antithetic_variates_mc(S0, K, r, sigma, T, n_paths, 'call', seed)
            
            st.markdown("**Comparação:**")
            
            comp_df = pd.DataFrame({
                'Método': ['MC Padrão', 'Antitético'],
                'Preço': [mc_std['price'], mc_anti['price']],
                'SE': [mc_std['se'], mc_anti['se']],
                'Erro vs BS': [abs(mc_std['price'] - bs_price), abs(mc_anti['price'] - bs_price)]
            })
            st.dataframe(comp_df.round(4), use_container_width=True, hide_index=True)
            
            reducao = (1 - mc_anti['se'] / mc_std['se']) * 100
            st.metric("Redução de SE", f"{reducao:.1f}%")
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Variáveis de Controle")
            
            st.markdown("""
            **Ideia:** Usar um problema correlacionado com solução conhecida.
            
            Se a variável de controle C tem E[C] conhecido:
            $$\\hat{X}_{adj} = \\hat{X} - \\beta(\\bar{C} - E[C])$$
            
            **Exemplo:** Usar preço do ativo (E[S_T·e^{-rT}] = S₀) 
            para ajustar estimativa da opção.
            
            **Ganho:** Depende da correlação entre payoff e controle.
            """)
        
        with col2:
            mc_ctrl = control_variate_mc(S0, K, r, sigma, T, n_paths, 'call', seed)
            
            st.markdown("**Comparação:**")
            
            comp_df = pd.DataFrame({
                'Método': ['MC Padrão', 'Controle'],
                'Preço': [mc_std['price'], mc_ctrl['price']],
                'SE': [mc_std['se'], mc_ctrl['se']],
                'Erro vs BS': [abs(mc_std['price'] - bs_price), abs(mc_ctrl['price'] - bs_price)]
            })
            st.dataframe(comp_df.round(4), use_container_width=True, hide_index=True)
            
            st.metric("Beta ótimo", f"{mc_ctrl['beta']:.3f}")
            reducao_ctrl = (1 - mc_ctrl['se'] / mc_std['se']) * 100
            st.metric("Redução de SE", f"{reducao_ctrl:.1f}%")
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Quasi-Monte Carlo")
            
            st.markdown("""
            **Ideia:** Em vez de sorteios aleatórios, usar sequências 
            de **baixa discrepância** que preenchem o espaço uniformemente.
            
            **Sequências comuns:**
            - Halton
            - Sobol
            - Niederreiter
            
            **Vantagem:** Convergência O(1/N) em vez de O(1/√N).
            
            **Desvantagem:** Não dá erro padrão tradicional.
            """)
        
        with col2:
            mc_qmc = quasi_mc_low_discrepancy(S0, K, r, sigma, T, n_paths, 'call')
            
            st.markdown("**Comparação:**")
            
            comp_df = pd.DataFrame({
                'Método': ['MC Padrão', 'Quasi-MC (Halton)'],
                'Preço': [mc_std['price'], mc_qmc['price']],
                'Erro vs BS': [abs(mc_std['price'] - bs_price), abs(mc_qmc['price'] - bs_price)]
            })
            st.dataframe(comp_df.round(4), use_container_width=True, hide_index=True)
    
    # Placar de eficiência
    st.subheader("📊 Placar de Eficiência")
    
    methods = ['MC Padrão', 'Antitético', 'Controle', 'Quasi-MC']
    errors = [abs(mc_std['price'] - bs_price), abs(mc_anti['price'] - bs_price),
              abs(mc_ctrl['price'] - bs_price), abs(mc_qmc['price'] - bs_price)]
    ses = [mc_std['se'], mc_anti['se'], mc_ctrl['se'], mc_qmc['se']]
    
    fig = go.Figure(data=[
        go.Bar(name='Erro vs BS', x=methods, y=errors),
        go.Bar(name='SE', x=methods, y=ses)
    ])
    fig.update_layout(barmode='group', height=300, title="Comparação de Métodos")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa antitéticas quando precisa de ganho rápido
    - Usa controle quando há proxy correlacionado
    - Considera QMC para problemas de alta dimensão
    """)


def render_section_S6():
    """S6: Bootstrapping: Aprendendo com os Próprios Dados"""
    st.header("🔄 Bootstrap: Inferência pelos Dados")
    
    st.markdown("""
    **Bootstrap:** Reamostra os próprios dados (com reposição) para estimar incerteza,
    **sem assumir distribuição teórica**.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Como Funciona")
        
        st.markdown("""
        **Algoritmo:**
        1. Dados originais: x₁, x₂, ..., xₙ
        2. Criar amostra bootstrap: sortear n valores **com reposição**
        3. Calcular estatística na amostra bootstrap
        4. Repetir B vezes (ex: 1000)
        5. Distribuição das estatísticas = incerteza
        
        **Vantagens:**
        - Não assume Normalidade
        - Funciona para qualquer estatística
        - Simples de implementar
        
        **IC Percentílico:**
        - Limite inferior: percentil 2.5%
        - Limite superior: percentil 97.5%
        """)
        
        # Gerar dados de exemplo
        n_obs = st.slider("Tamanho da amostra original", 20, 200, 50, key="n_boot")
        n_bootstrap = st.slider("Número de reamostras", 100, 5000, 1000, key="b_boot")
        
        seed_boot = st.number_input("Seed", 1, 9999, 42, key="seed_boot")
    
    with col2:
        # Gerar dados (não-normais para mostrar valor do bootstrap)
        np.random.seed(seed_boot)
        data = np.random.exponential(10, n_obs)  # Assimétrica
        
        st.markdown("**Dados originais (exponencial):**")
        st.metric("Média amostral", f"{np.mean(data):.2f}")
        st.metric("Mediana amostral", f"{np.median(data):.2f}")
        
        # Bootstrap para média
        boot_means = bootstrap_resample(data, n_bootstrap, np.mean, seed_boot)
        boot_medians = bootstrap_resample(data, n_bootstrap, np.median, seed_boot)
        
        ci_mean = bootstrap_ci(boot_means, 0.95)
        ci_median = bootstrap_ci(boot_medians, 0.95)
    
    # Visualização
    st.subheader("Distribuição Bootstrap")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(boot_means, nbins=50, title="Bootstrap da Média")
        fig.add_vline(x=ci_mean['lower'], line_dash="dash", line_color="red")
        fig.add_vline(x=ci_mean['upper'], line_dash="dash", line_color="red")
        fig.add_vline(x=np.mean(data), line_dash="solid", line_color="green",
                     annotation_text="Estimativa pontual")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("IC 95% (Média)", f"[{ci_mean['lower']:.2f}, {ci_mean['upper']:.2f}]")
    
    with col2:
        fig = px.histogram(boot_medians, nbins=50, title="Bootstrap da Mediana")
        fig.add_vline(x=ci_median['lower'], line_dash="dash", line_color="red")
        fig.add_vline(x=ci_median['upper'], line_dash="dash", line_color="red")
        fig.add_vline(x=np.median(data), line_dash="solid", line_color="green",
                     annotation_text="Estimativa pontual")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("IC 95% (Mediana)", f"[{ci_median['lower']:.2f}, {ci_median['upper']:.2f}]")
    
    # Comparação com IC paramétrico
    with st.expander("📖 Comparação: Bootstrap vs Paramétrico"):
        # IC paramétrico para média (assume normalidade)
        se_parametrico = np.std(data, ddof=1) / np.sqrt(n_obs)
        ci_param = (np.mean(data) - 1.96 * se_parametrico, 
                   np.mean(data) + 1.96 * se_parametrico)
        
        st.markdown(f"""
        **IC 95% para Média:**
        - Paramétrico (assume Normal): [{ci_param[0]:.2f}, {ci_param[1]:.2f}]
        - Bootstrap (não assume): [{ci_mean['lower']:.2f}, {ci_mean['upper']:.2f}]
        
        **Diferença:** O IC paramétrico assume que a média amostral é Normal.
        Para dados assimétricos (como exponencial), bootstrap pode ser mais preciso.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa bootstrap quando não quer/pode assumir Normalidade
    - Reporta IC bootstrap junto com estimativa pontual
    - Funciona para estatísticas complexas (Sharpe, VaR, etc.)
    """)


def render_section_S7():
    """S7: Bootstrap em Risco: VaR e Capital (visão MBA)"""
    st.header("📉 Bootstrap para VaR e Capital")
    
    st.markdown("""
    Bootstrap é especialmente útil para **medidas de risco** onde caudas importam
    e assumir Normalidade pode ser perigoso.
    """)
    
    tab1, tab2 = st.tabs(["📊 VaR Paramétrico vs Bootstrap", "✅ Checklist"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Configuração")
            
            distribution = st.radio(
                "Distribuição dos retornos:",
                ['normal', 't'],
                format_func=lambda x: 'Normal' if x == 'normal' else 't-Student (caudas pesadas)',
                key="dist_var"
            )
            
            n_returns = st.slider("Dias de histórico", 100, 1000, 252, key="n_var")
            confidence = st.slider("Nível de confiança VaR", 0.90, 0.99, 0.95, 0.01, key="conf_var")
            n_bootstrap_var = st.slider("Reamostras bootstrap", 500, 5000, 1000, key="b_var")
            
            seed_var = st.number_input("Seed", 1, 9999, 42, key="seed_var")
        
        with col2:
            # Gerar retornos
            if distribution == 'normal':
                np.random.seed(seed_var)
                returns = np.random.normal(0, 0.02, n_returns)
            else:
                np.random.seed(seed_var)
                returns = np.random.standard_t(4, n_returns) * 0.02 / np.sqrt(4 / 2)
            
            # VaR paramétrico (assume Normal)
            var_param = -stats.norm.ppf(1 - confidence) * np.std(returns)
            
            # VaR histórico
            var_hist = -np.percentile(returns, (1 - confidence) * 100)
            
            # VaR bootstrap
            var_boot = var_bootstrap(returns, confidence, n_bootstrap_var, seed_var)
            
            st.markdown("**Comparação de Métodos:**")
            
            results_df = pd.DataFrame({
                'Método': ['Paramétrico (Normal)', 'Histórico', 'Bootstrap (média)'],
                'VaR': [f"{var_param*100:.2f}%", f"{var_hist*100:.2f}%", 
                       f"{var_boot['var_mean']*100:.2f}%"],
                'IC 95%': ['N/A', 'N/A', 
                          f"[{var_boot['var_lower']*100:.2f}%, {var_boot['var_upper']*100:.2f}%]"]
            })
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Distribuição bootstrap do VaR
        st.subheader("Distribuição Bootstrap do VaR")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(var_boot['bootstrap_vars'] * 100, nbins=50,
                              labels={'value': 'VaR (%)', 'count': 'Frequência'},
                              title="Incerteza no VaR (Bootstrap)")
            fig.add_vline(x=var_param * 100, line_dash="dash", line_color="red",
                         annotation_text="VaR Paramétrico")
            fig.add_vline(x=var_boot['var_mean'] * 100, line_dash="solid", line_color="green",
                         annotation_text="VaR Bootstrap")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Impacto no capital
            capital_param = var_param * 1000000  # Assumindo portfolio de 1M
            capital_boot = var_boot['var_mean'] * 1000000
            
            st.markdown("**Impacto no Capital (portfolio R$ 1M):**")
            st.metric("Capital (Paramétrico)", f"R$ {capital_param:,.0f}")
            st.metric("Capital (Bootstrap)", f"R$ {capital_boot:,.0f}",
                     delta=f"R$ {capital_boot - capital_param:,.0f}")
            
            if distribution == 't' and capital_boot > capital_param:
                st.warning("⚠️ Com caudas pesadas, VaR paramétrico subestima capital!")
    
    with tab2:
        st.subheader("Quando Usar Bootstrap para Risco?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ Recomendado:**
            - Retornos não-normais
            - Caudas pesadas
            - Amostras pequenas
            - Estatísticas complexas (ES, drawdown)
            - Quando incerteza do VaR importa
            """)
        
        with col2:
            st.markdown("""
            **⚠️ Cuidados:**
            - Precisa de dados suficientes (>100 obs)
            - Assume estacionaridade
            - Não captura mudanças de regime
            - Computacionalmente mais intenso
            """)
        
        st.info("""
        💡 **Regra prática:** Use bootstrap quando não confia na Normalidade
        ou quando quer reportar incerteza da medida de risco, não só o ponto.
        """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Reporta VaR com intervalo de confiança (não só ponto)
    - Usa bootstrap quando caudas são pesadas
    - Considera a incerteza do VaR nas decisões de capital
    """)


def render_section_S8():
    """S8: Estudo de Caso MBA e Limitações Estratégicas"""
    st.header("💼 Estudo de Caso: Simulação de Portfólio")
    
    st.markdown("""
    Vamos simular o risco de um portfólio e discutir alertas importantes.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuração do Portfólio")
        
        st.markdown("""
        **Portfólio com 3 ativos:**
        - Ação A: μ=12%, σ=20%
        - Ação B: μ=8%, σ=15%
        - Renda Fixa: μ=5%, σ=5%
        """)
        
        w_a = st.slider("Peso Ação A (%)", 0, 100, 40, key="w_a")
        w_b = st.slider("Peso Ação B (%)", 0, 100 - w_a, 30, key="w_b")
        w_rf = 100 - w_a - w_b
        
        st.markdown(f"**Peso Renda Fixa:** {w_rf}%")
        
        n_sim_port = st.slider("Simulações", 1000, 50000, 10000, key="n_port")
        
        use_seed = st.checkbox("Fixar seed para reprodutibilidade", value=True, key="fix_seed")
        if use_seed:
            seed_port = st.number_input("Seed", 1, 9999, 42, key="seed_port")
        else:
            seed_port = None
    
    with col2:
        # Simular
        weights = np.array([w_a, w_b, w_rf]) / 100
        
        # Gerar retornos históricos sintéticos
        np.random.seed(42)
        mean_returns = np.array([0.12, 0.08, 0.05]) / 252
        cov_matrix = np.array([
            [0.20**2, 0.15**2 * 0.5, 0.20 * 0.05 * 0.1],
            [0.15**2 * 0.5, 0.15**2, 0.15 * 0.05 * 0.2],
            [0.20 * 0.05 * 0.1, 0.15 * 0.05 * 0.2, 0.05**2]
        ]) / 252
        
        returns_matrix = np.random.multivariate_normal(mean_returns, cov_matrix, 252)
        
        # Simular portfólio
        result = case_portfolio_sim(weights, returns_matrix, n_sim_port, seed_port)
        
        st.subheader("Métricas de Risco")
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("VaR 95% (diário)", f"{result['var_95']*100:.2f}%")
        col_m2.metric("ES 95%", f"{result['es_95']*100:.2f}%")
        
        col_m3, col_m4 = st.columns(2)
        col_m3.metric("Retorno médio (diário)", f"{result['mean']*100:.4f}%")
        col_m4.metric("Sharpe (anualizado)", f"{result['sharpe']:.2f}")
    
    # Distribuição
    st.subheader("Distribuição dos Retornos Simulados")
    
    fig = px.histogram(result['portfolio_returns'] * 100, nbins=100,
                      labels={'value': 'Retorno (%)', 'count': 'Frequência'},
                      title="Distribuição do Retorno Diário do Portfólio")
    fig.add_vline(x=-result['var_95'] * 100, line_dash="dash", line_color="red",
                 annotation_text=f"VaR 95%: {result['var_95']*100:.2f}%")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    # Log de parâmetros
    if use_seed:
        params = {
            'Pesos': f"A={w_a}%, B={w_b}%, RF={w_rf}%",
            'N simulações': n_sim_port,
            'Seed': seed_port
        }
        log = log_params_and_seed(params, seed_port)
        
        with st.expander("📋 Log de Parâmetros (para reprodutibilidade)"):
            st.code(log)
    
    # Alertas
    st.subheader("⚠️ Alertas e Limitações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Snooping:**
        - Testar muitas estratégias até achar uma "boa"
        - Resultado pode ser sorte, não skill
        - Solução: Out-of-sample testing
        
        **Viés de Sobrevivência:**
        - Usar só dados de empresas que existem hoje
        - Ignora falências → superestima retornos
        - Solução: Usar dados com delisted
        """)
    
    with col2:
        st.markdown("""
        **Premissas do DGP:**
        - Assumir Normal quando caudas são pesadas
        - Ignorar mudanças de regime
        - Correlações constantes
        
        **Custo Computacional:**
        - Mais precisão = mais tempo
        - Trade-off custo-benefício
        - Paralelização quando possível
        """)
    
    st.error("""
    🚨 **Regra de ouro:** Simulação é tão boa quanto suas premissas.
    Sempre documente, teste robustez e seja cético com resultados "bons demais".
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Documenta todos os parâmetros e seeds
    - Testa robustez com diferentes premissas
    - Faz out-of-sample antes de implementar
    - Desconfia de resultados muito bons
    """)


def render_section_S9():
    """S9: Resumo Executivo e Encerramento"""
    st.header("📋 Resumo Executivo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### O que Aprendemos sobre Simulação
        
        ✅ **Quando Simular:**
        - Fórmulas fechadas insuficientes
        - Premissas de modelos frágeis
        - Quer entender distribuição completa
        - "Test-drive" antes de implementar
        
        ✅ **Monte Carlo:**
        - Repetir sorteios para estimar resultados
        - Erro cai como 1/√N
        - 10x menos erro = 100x mais simulações
        
        ✅ **Aplicações Financeiras:**
        - Precificação de opções (trajetórias GBM)
        - Análise de risco (VaR, ES)
        - Validação de estratégias
        
        ✅ **Caudas Pesadas:**
        - Normal subestima eventos extremos
        - Usar t-Student ou GARCH
        - Impacto significativo no capital
        
        ✅ **Redução de Variância:**
        - Antitéticas: pares (z, -z)
        - Controle: usar proxy conhecido
        - Quasi-MC: sequências de baixa discrepância
        
        ✅ **Bootstrap:**
        - Reamostrar com reposição
        - Não assume Normalidade
        - IC para qualquer estatística
        
        ✅ **Governança:**
        - Fixar seed para reprodutibilidade
        - Documentar parâmetros
        - Testar robustez
        - Cuidado com data snooping
        """)
    
    with col2:
        st.markdown("### 💡 Mensagem-Chave")
        
        st.info("""
        **"Simulação é um laboratório para decisões sob incerteza"**
        
        Permite:
        - Testar antes de arriscar
        - Quantificar incerteza
        - Explorar cenários extremos
        - Validar intuição
        
        Mas exige:
        - Premissas realistas
        - Documentação
        - Ceticismo saudável
        """)
        
        st.markdown("### 🧪 Quiz Final")
        
        resposta = st.radio(
            "Para reduzir o erro de MC pela metade, preciso:",
            ["2x mais simulações",
             "4x mais simulações",
             "10x mais simulações"],
            key="quiz_final"
        )
        
        if st.button("Ver resposta", key="btn_final"):
            if resposta == "4x mais simulações":
                st.success("""
                ✅ **Correto!**
                
                Erro ∝ 1/√N
                
                Para reduzir pela metade:
                1/2 = 1/√(N_novo/N_antigo)
                √(N_novo/N_antigo) = 2
                N_novo/N_antigo = 4
                """)
            else:
                st.error("Lembre: erro ∝ 1/√N. Para erro/2, precisa 4x mais simulações.")
    
    st.markdown("---")
    
    st.subheader("🎓 Encerramento do Módulo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Ferramentas:**
        - Monte Carlo
        - Bootstrap
        - Redução de variância
        - Quasi-MC
        """)
    
    with col2:
        st.markdown("""
        **Aplicações:**
        - Precificação
        - VaR/ES
        - Backtesting
        - Validação
        """)
    
    with col3:
        st.markdown("""
        **Alertas:**
        - Data snooping
        - Sobrevivência
        - Premissas
        - Reprodutibilidade
        """)
    
    st.success("""
    🎓 **Simulação completa o toolkit do MBA em Econometria!**
    
    Com simulação, você pode:
    - Explorar "e se..." sem risco real
    - Quantificar incerteza de forma robusta
    - Validar estratégias antes de implementar
    - Comunicar risco com confiança
    
    **Pratique com seus próprios dados e problemas!**
    """)
    
    st.markdown("""
    ---
    **🎯 O que um gestor faz com isso?**
    - Usa simulação como ferramenta de planejamento
    - Documenta e torna reprodutível
    - Combina com outras técnicas do curso
    - Toma decisões informadas sob incerteza
    """)


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título e objetivos
    st.title("🎲 Módulo 11: Métodos de Simulação")
    st.markdown("**Laboratório de Econometria** | Monte Carlo, Bootstrap e Aplicações")
    
    with st.expander("🎯 Objetivos do Módulo", expanded=False):
        st.markdown("""
        - Explicar **simulação** como ferramenta de decisão
        - Ensinar **Monte Carlo** e convergência com N
        - Aplicar em **precificação de opções** e **análise de risco**
        - Mostrar impacto de **caudas pesadas**
        - Introduzir **técnicas de redução de variância**
        - Ensinar **bootstrap** para inferência robusta
        - Consolidar com **caso MBA** e alertas de governança
        """)
    
    # Sidebar: navegação
    st.sidebar.title("📑 Navegação")
    
    secoes = {
        "S1": "🎲 Lógica da Simulação",
        "S2": "🎰 Monte Carlo",
        "S3": "📈 Precificação de Opções",
        "S4": "📊 Caudas Pesadas",
        "S5": "⚡ Redução de Variância",
        "S6": "🔄 Bootstrap",
        "S7": "📉 VaR Bootstrap",
        "S8": "💼 Caso MBA",
        "S9": "📋 Resumo"
    }
    
    secao_selecionada = st.sidebar.radio(
        "Selecione a seção:",
        list(secoes.keys()),
        format_func=lambda x: secoes[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Dica:** Simulação é o 
    "laboratório" do gestor para 
    testar decisões sob incerteza.
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
            page_title="Módulo 11: Métodos de Simulação",
            page_icon="🎲",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()