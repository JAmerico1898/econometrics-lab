"""
Laboratório de Econometria - Aplicativo Principal
"""

import streamlit as st

# Configuração da página (deve ser a primeira chamada Streamlit)
st.set_page_config(
    page_title="Laboratório de Econometria",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Importação dos módulos
import module_01_statistics_review
import module_02_classical_linear_regression
import module_03_further_clrm
import module_04_clrm_diagnostics
import module_05_causality_identification
import module_06_univariate_time_series
import module_07_multivariate_models
import module_08_long_run_relationships
import module_09_volatility_correlation
import module_10_panel_data
import module_11_simulation_methods
import module_12_suggestions

#CABEÇALHO DO FORM
st.markdown("<h2 style='text-align: center;'>Laboratório de Econometria</h2>", unsafe_allow_html=True)

st.markdown("<hr style='border:0.5px solid black;'>", unsafe_allow_html=True)

# Define your options
options = [
            "M1 - Revisão de Estatística",
            "M2 - Regressão Linear",
            "M3 - Desenvolvimentos Adicionais - Regressão Linear", 
            "M4 - Pressupostos e Teste Diagnósticos - Regressão Linear",
            "M5 - Causalidade e Identificação",
            "M6 - Modelagem de Séries Temporais e Forecasting",
            "M7 - Modelos Multivariados",
            "M8 - Modelando Relações de Longo Prazo em Finanças",
            "M9 - Modelando Volatilidade e Correlação",
            "M10 - Dados em Painel",
            "M11 - Métodos de Simulação",
            "Caixa de Sugestões, Dúvidas...!"
]

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

# Initialize session state variables if they don't exist
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
    st.session_state.should_scroll = False

# Define button click handlers for each option
def select_option(option):
    st.session_state.selected_option = option

# Define custom CSS for button styling
st.markdown("""
<style>
    /* Default button style (light gray) */
    .stButton > button {
        background-color: #f0f2f6 !important;
        color: #31333F !important;
        border-color: #d2d6dd !important;
        width: 100%;
    }
    
    /* Selected button style (red) */
    .selected-button {
        background-color: #FF4B4B !important;
        color: white !important;
        border-color: #FF0000 !important;
        width: 100%;
        padding: 0.5rem;
        font-weight: 400;
        border-radius: 0.25rem;
        cursor: default;
        text-align: center;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Create three columns for the initial choices
col1, col2, col3 = st.columns([3, 3, 3])

# Row 1
with col1:
    if st.session_state.selected_option == options[0]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[0]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[0], key="btn0", use_container_width=True, on_click=select_option, args=(options[0],))

with col2:
    if st.session_state.selected_option == options[1]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[1]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[1], key="btn1", use_container_width=True, on_click=select_option, args=(options[1],))

with col3:
    if st.session_state.selected_option == options[2]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[2]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[2], key="btn2", use_container_width=True, on_click=select_option, args=(options[2],))


# Create three columns for the initial choices
col4, col5, col6 = st.columns([3, 3, 3])

# Row 2
with col4:
    if st.session_state.selected_option == options[3]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[3]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[3], key="btn3", use_container_width=True, on_click=select_option, args=(options[3],))

with col5:
    if st.session_state.selected_option == options[4]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[4]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[4], key="btn4", use_container_width=True, on_click=select_option, args=(options[4],))

with col6:
    if st.session_state.selected_option == options[5]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[5]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[5], key="btn5", use_container_width=True, on_click=select_option, args=(options[5],))

# Create three columns for the initial choices
col7, col8, col9 = st.columns([3, 3, 3])

# Row 3
with col7:
    if st.session_state.selected_option == options[6]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[6]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[6], key="btn6", use_container_width=True, on_click=select_option, args=(options[6],))

with col8:
    if st.session_state.selected_option == options[7]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[7]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[7], key="btn7", use_container_width=True, on_click=select_option, args=(options[7],))

with col9:
    if st.session_state.selected_option == options[8]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[8]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )

    else:
        # Display default (gray) button
        st.button(options[8], key="btn8", use_container_width=True, on_click=select_option, args=(options[8],))


# Create three columns for the initial choices
col10, col11, col12 = st.columns([3, 3, 3])

# Row 4
with col10:
    if st.session_state.selected_option == options[9]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[9]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[9], key="btn9", use_container_width=True, on_click=select_option, args=(options[9],))

with col11:
    if st.session_state.selected_option == options[10]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[10]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[10], key="btn10", use_container_width=True, on_click=select_option, args=(options[10],))
        
with col12:
    if st.session_state.selected_option == options[11]:
        # Display selected (red) button
        st.markdown(
            f"""
            <div data-testid="stButton">
                <button class="selected-button">
                    {options[11]}
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Display default (gray) button
        st.button(options[11], key="btn11", use_container_width=True, on_click=select_option, args=(options[11],))
        
        
st.markdown("<hr style='border:0.5px solid black;'>", unsafe_allow_html=True)


# =============================================================================
# RENDERIZAÇÃO DOS MÓDULOS
# =============================================================================

if st.session_state.selected_option == "M1 - Revisão de Estatística":
    module_01_statistics_review.render()

elif st.session_state.selected_option == "M2 - Regressão Linear":
    module_02_classical_linear_regression.render()
    
elif st.session_state.selected_option == "M3 - Desenvolvimentos Adicionais - Regressão Linear":
    module_03_further_clrm.render()

elif st.session_state.selected_option == "M4 - Pressupostos e Teste Diagnósticos - Regressão Linear":
    module_04_clrm_diagnostics.render()

elif st.session_state.selected_option == "M5 - Causalidade e Identificação":
    module_05_causality_identification.render()

elif st.session_state.selected_option == "M6 - Modelagem de Séries Temporais e Forecasting":
    module_06_univariate_time_series.render()

elif st.session_state.selected_option == "M7 - Modelos Multivariados":
    module_07_multivariate_models.render()

elif st.session_state.selected_option == "M8 - Modelando Relações de Longo Prazo em Finanças":
    module_08_long_run_relationships.render()

elif st.session_state.selected_option == "M9 - Modelando Volatilidade e Correlação":
    module_09_volatility_correlation.render()

elif st.session_state.selected_option == "M10 - Dados em Painel":
    module_10_panel_data.render()

elif st.session_state.selected_option == "M11 - Métodos de Simulação":
    module_11_simulation_methods.render()

elif st.session_state.selected_option == "Caixa de Sugestões, Dúvidas...!":
    module_12_suggestions.render()

else:
    # Nenhum módulo selecionado - mostrar mensagem de boas-vindas
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #666;'>
        <h3>👆 Selecione um módulo acima para começar</h3>
        <p>Clique em um dos botões para acessar o conteúdo interativo.</p>
    </div>
    """, unsafe_allow_html=True)
    
# -----------------------------------------------------------------------------
# RODAPÉ
# -----------------------------------------------------------------------------
# Footer
st.divider()

# Rodapé
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    📊 © 2026 Laboratório de Econometria | Desenvolvido para fins educacionais<br>
    Prof. José Américo – Coppead - FGV - UCAM
</div>
""", unsafe_allow_html=True)
