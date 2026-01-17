# 📊 Laboratório de Econometria

**Aplicativo educacional interativo para ensino de Econometria aplicada a negócios**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Sobre o Projeto

O **Laboratório de Econometria** é uma aplicação Streamlit desenvolvida para alunos de MBA em Economia, Finanças e Gestão. O objetivo é maximizar a **intuição gerencial**, **aplicação prática** e **clareza didática**, evitando excessos matemáticos.

### Filosofia Pedagógica

- **"Show, don't tell"**: Simulações interativas antes da teoria
- **Foco gerencial**: "O que um gestor faz com isso?" ao final de cada seção
- **Linguagem acessível**: Português com termos técnicos em inglês quando usuais no mercado
- **Aprendizado ativo**: Quizzes com feedback imediato em cada módulo

---

## 📚 Módulos

| Módulo | Tema | Conteúdo Principal |
|--------|------|-------------------|
| **1** | Revisão de Estatística | Distribuições, TLC, inferência, IC, testes de hipótese |
| **2** | Modelo de Regressão Linear Clássico | OLS, pressupostos CLRM, interpretação de coeficientes |
| **3** | Diagnóstico do Modelo | Heterocedasticidade, autocorrelação, multicolinearidade |
| **4** | Correções e Extensões | Erros robustos (HC, HAC), GLS, variáveis instrumentais |
| **5** | Causalidade e Identificação | RCT, diff-in-diff, RDD, variáveis instrumentais |
| **6** | Séries Temporais Univariadas | Estacionaridade, ARIMA, previsão, sazonalidade |
| **7** | Modelos Multivariados | SEM, IV/2SLS, VAR, IRF, causalidade de Granger |
| **8** | Relações de Longo Prazo | Cointegração, ECM, VECM, teste de Johansen |
| **9** | Volatilidade e Correlação | GARCH, assimetria, DCC, VaR, aplicações em risco |
| **10** | Dados em Painel | FE, RE, teste de Hausman, SUR, cointegração em painel |
| **11** | Métodos de Simulação | Monte Carlo, bootstrap, redução de variância, VaR |

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)

### Passos

1. **Clone ou baixe os arquivos**

```bash
# Criar diretório do projeto
mkdir laboratorio-econometria
cd laboratorio-econometria
```

2. **Instale as dependências**

```bash
pip install -r requirements.txt
```

3. **Execute a aplicação**

```bash
streamlit run econometrics_lab.py
```

4. **Acesse no navegador**

```
http://localhost:8501
```

---

## 📁 Estrutura de Arquivos

```
laboratorio-econometria/
├── econometrics_lab.py                    # Aplicação principal
├── requirements.txt                        # Dependências
├── README.md                              # Este arquivo
│
├── module_01_statistics_review.py         # Módulo 1: Estatística
├── module_02_classical_linear_regression.py # Módulo 2: CLRM
├── module_03_model_diagnostics.py         # Módulo 3: Diagnóstico
├── module_04_corrections_extensions.py    # Módulo 4: Correções
├── module_05_causality_identification.py  # Módulo 5: Causalidade
├── module_06_univariate_time_series.py    # Módulo 6: Séries Univariadas
├── module_07_multivariate_models.py       # Módulo 7: Multivariados
├── module_08_long_run_relationships.py    # Módulo 8: Longo Prazo
├── module_09_volatility_correlation.py    # Módulo 9: Volatilidade
├── module_10_panel_data.py                # Módulo 10: Painel
└── module_11_simulation_methods.py        # Módulo 11: Simulação
```

---

## 🎮 Como Usar

### Navegação

1. **Selecione o módulo** na barra lateral esquerda
2. **Escolha a seção** dentro do módulo usando o menu de navegação
3. **Interaja** com sliders, botões e controles para explorar os conceitos
4. **Responda os quizzes** para testar seu entendimento

### Recursos Interativos

- **Sliders**: Ajuste parâmetros e veja os efeitos em tempo real
- **Tabs**: Compare diferentes métodos ou cenários
- **Expanders**: Acesse notas técnicas opcionais para aprofundamento
- **Métricas**: Visualize números-chave destacados
- **Gráficos Plotly**: Interaja com zoom, hover e seleção

---

## 📖 Conteúdo por Módulo

### Módulo 1: Revisão de Estatística
- Distribuições (Normal, t, χ², F)
- Teorema Central do Limite
- Intervalos de confiança
- Testes de hipótese
- Erros Tipo I e II

### Módulo 2: Regressão Linear Clássica
- Intuição do OLS
- Pressupostos CLRM
- Interpretação de coeficientes
- R² e ajuste do modelo
- Inferência e testes t/F

### Módulo 3: Diagnóstico do Modelo
- Heterocedasticidade (Breusch-Pagan, White)
- Autocorrelação (Durbin-Watson, Breusch-Godfrey)
- Multicolinearidade (VIF)
- Normalidade dos resíduos
- Especificação (RESET)

### Módulo 4: Correções e Extensões
- Erros robustos HC0-HC3
- Erros HAC (Newey-West)
- Mínimos Quadrados Generalizados
- Variáveis instrumentais (introdução)

### Módulo 5: Causalidade e Identificação
- Correlação vs causalidade
- Experimentos aleatorizados (RCT)
- Diferenças-em-diferenças
- Regressão descontínua (RDD)
- Variáveis instrumentais

### Módulo 6: Séries Temporais Univariadas
- Estacionaridade
- Processos AR, MA, ARMA, ARIMA
- Testes de raiz unitária (ADF, KPSS)
- Previsão e validação
- Sazonalidade

### Módulo 7: Modelos Multivariados
- Viés de simultaneidade
- Forma estrutural vs reduzida
- Identificação
- IV e 2SLS
- VAR, IRF, FEVD
- Causalidade de Granger

### Módulo 8: Relações de Longo Prazo
- Regressão espúria
- Cointegração
- ECM (Error Correction Model)
- VECM
- Teste de Johansen

### Módulo 9: Volatilidade e Correlação
- Fatos estilizados de retornos
- Volatilidade histórica vs EWMA
- GARCH(1,1)
- Modelos assimétricos (GJR, EGARCH)
- DCC e correlação dinâmica
- VaR e aplicações em risco

### Módulo 10: Dados em Painel
- Estrutura de painel
- Pooled OLS
- Efeitos Fixos (FE)
- Efeitos Aleatórios (RE)
- Teste de Hausman
- SUR
- Cointegração em painel

### Módulo 11: Métodos de Simulação
- Lógica da simulação
- Monte Carlo e convergência
- Precificação de opções
- Caudas pesadas e risco
- Redução de variância
- Bootstrap
- VaR via simulação

---

## 🛠️ Dependências

| Pacote | Versão | Uso |
|--------|--------|-----|
| streamlit | ≥1.28.0 | Framework da aplicação |
| pandas | ≥2.0.0 | Manipulação de dados |
| numpy | ≥1.24.0 | Cálculos numéricos |
| scipy | ≥1.10.0 | Estatística e testes |
| plotly | ≥5.15.0 | Visualizações interativas |

---

## 🎓 Público-Alvo

- **Alunos de MBA** em Economia, Finanças e Gestão
- **Profissionais** que querem revisar conceitos de econometria
- **Gestores** que precisam interpretar análises quantitativas
- **Analistas** que buscam intuição prática sobre métodos estatísticos

---

## ✨ Características

- ✅ **100% interativo**: Todos os conceitos com simulações
- ✅ **Auto-contido**: Não requer arquivos de dados externos
- ✅ **Dados sintéticos**: Gerados internamente com parâmetros ajustáveis
- ✅ **Reprodutível**: Controle de seed para replicação
- ✅ **Responsivo**: Interface adaptável a diferentes tamanhos de tela
- ✅ **Em português**: Linguagem acessível com termos técnicos preservados

---

## 📝 Licença

Este projeto é disponibilizado para fins educacionais.

---

## 👨‍🏫 Créditos

Desenvolvido para o curso de Econometria do MBA COPPEAD/UFRJ.

---

## 🐛 Problemas Conhecidos

- Em alguns navegadores, gráficos Plotly podem demorar a carregar na primeira vez
- Sliders com muitas simulações podem causar lentidão temporária

---

## 📧 Contato

Para dúvidas ou sugestões sobre o conteúdo pedagógico, entre em contato com o professor responsável.

---

**Bons estudos! 📈**