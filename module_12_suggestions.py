"""
Laboratório de Econometria - Module 12: Caixa de Sugestões
Canal de comunicação para dúvidas, sugestões e feedback dos alunos.
Envia notificações via Pushover para o professor.
"""

import streamlit as st
import requests
from datetime import datetime

# =============================================================================
# CONFIGURAÇÃO DO PUSHOVER
# =============================================================================

try:
    PUSHOVER_USER_KEY = st.secrets.get("PUSHOVER_USER_KEY", "")
    PUSHOVER_API_TOKEN = st.secrets.get("PUSHOVER_API_TOKEN", "")
except Exception as e:
    PUSHOVER_USER_KEY = ""
    PUSHOVER_API_TOKEN = ""


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def send_contact_to_admin(user_name: str, user_email: str, category: str, 
                          message: str, module: str = "Geral") -> bool:
    """Envia mensagem para o administrador via Pushover."""
    if not PUSHOVER_USER_KEY or not PUSHOVER_API_TOKEN:
        return False
    
    try:
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        pushover_message = f"""📬 Nova mensagem - Laboratório de Econometria

📅 Data: {timestamp}
👤 Nome: {user_name}
📧 E-mail: {user_email}
📂 Categoria: {category}
📚 Módulo: {module}

💬 Mensagem:
{message}"""
        
        # Definir prioridade baseada na categoria
        priority = 0  # Normal
        if category == "🚨 Erro/Bug no aplicativo":
            priority = 1  # Alta
        
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_API_TOKEN,
                "user": PUSHOVER_USER_KEY,
                "message": pushover_message,
                "title": f"Lab Econometria - {category}",
                "priority": priority,
                "sound": "pushover"
            },
            timeout=10
        )
        
        return response.status_code == 200
    
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout ao enviar mensagem. Tente novamente.")
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erro de conexão: {e}")
        return False
    except Exception as e:
        st.error(f"❌ Erro ao enviar notificação: {e}")
        return False


def validate_email(email: str) -> bool:
    """Validação simples de formato de e-mail."""
    if not email:
        return False
    if "@" not in email or "." not in email:
        return False
    if len(email) < 5:
        return False
    return True


def validate_message(message: str, min_length: int = 10) -> bool:
    """Valida se a mensagem tem conteúdo mínimo."""
    if not message:
        return False
    if len(message.strip()) < min_length:
        return False
    return True


# =============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =============================================================================

def render():
    """Função principal que renderiza o módulo completo."""
    
    # Título
    st.title("📬 Caixa de Sugestões")
    st.markdown("**Laboratório de Econometria** | Dúvidas, Sugestões e Feedback")
    
    st.markdown("---")
    
    # Introdução
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Fale com o Professor
        
        Use este canal para:
        - 🤔 **Tirar dúvidas** sobre o conteúdo dos módulos
        - 💡 **Enviar sugestões** de melhorias no aplicativo
        - 🐛 **Reportar erros** ou problemas técnicos
        - 📝 **Dar feedback** sobre sua experiência de aprendizado
        - 💬 **Fazer comentários** gerais sobre o curso
        
        Sua mensagem será enviada diretamente para o professor responsável.
        """)
    
    with col2:
        st.info("""
        💡 **Dica**
        
        Seja específico na sua mensagem!
        
        Inclua o módulo e seção 
        relevantes para facilitar 
        o atendimento.
        """)
    
    st.markdown("---")
    
    # Verificar configuração do Pushover
    pushover_configured = bool(PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN)
    
    if not pushover_configured:
        st.warning("""
        ⚠️ **Sistema de notificação não configurado**
        
        O envio de mensagens está temporariamente indisponível.
        Entre em contato com o professor por e-mail.
        """)
    
    # Formulário
    st.subheader("📝 Envie sua Mensagem")
    
    with st.form(key="contact_form", clear_on_submit=True):
        
        # Dados do usuário
        col1, col2 = st.columns(2)
        
        with col1:
            user_name = st.text_input(
                "👤 Seu nome *",
                placeholder="Digite seu nome",
                max_chars=100
            )
        
        with col2:
            user_email = st.text_input(
                "📧 Seu e-mail *",
                placeholder="seu.email@exemplo.com",
                max_chars=100
            )
        
        # Categoria e módulo
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox(
                "📂 Categoria *",
                options=[
                    "🤔 Dúvida sobre conteúdo",
                    "💡 Sugestão de melhoria",
                    "🚨 Erro/Bug no aplicativo",
                    "📝 Feedback geral",
                    "💬 Outro assunto"
                ]
            )
        
        with col2:
            module = st.selectbox(
                "📚 Módulo relacionado",
                options=[
                    "Geral / Não se aplica",
                    "Módulo 1 - Revisão de Estatística",
                    "Módulo 2 - Regressão Linear Clássica",
                    "Módulo 3 - Diagnóstico do Modelo",
                    "Módulo 4 - Correções e Extensões",
                    "Módulo 5 - Causalidade e Identificação",
                    "Módulo 6 - Séries Temporais Univariadas",
                    "Módulo 7 - Modelos Multivariados",
                    "Módulo 8 - Relações de Longo Prazo",
                    "Módulo 9 - Volatilidade e Correlação",
                    "Módulo 10 - Dados em Painel",
                    "Módulo 11 - Métodos de Simulação"
                ]
            )
        
        # Mensagem
        message = st.text_area(
            "💬 Sua mensagem *",
            placeholder="Descreva sua dúvida, sugestão ou feedback em detalhes...\n\nSe for uma dúvida, inclua:\n- O que você está tentando entender\n- O que já tentou\n- Onde encontrou dificuldade",
            height=200,
            max_chars=2000
        )
        
        # Contador de caracteres
        char_count = len(message) if message else 0
        st.caption(f"{char_count}/2000 caracteres")
        
        # Botão de envio
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            submitted = st.form_submit_button(
                "📤 Enviar Mensagem",
                use_container_width=True,
                type="primary"
            )
    
    # Processamento do envio
    if submitted:
        # Validações
        errors = []
        
        if not user_name or len(user_name.strip()) < 2:
            errors.append("Por favor, informe seu nome.")
        
        if not validate_email(user_email):
            errors.append("Por favor, informe um e-mail válido.")
        
        if not validate_message(message, min_length=10):
            errors.append("A mensagem deve ter pelo menos 10 caracteres.")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            if not pushover_configured:
                st.error("❌ Sistema de envio não configurado. Tente novamente mais tarde.")
            else:
                # Enviar mensagem
                with st.spinner("📤 Enviando mensagem..."):
                    success = send_contact_to_admin(
                        user_name=user_name.strip(),
                        user_email=user_email.strip(),
                        category=category,
                        message=message.strip(),
                        module=module
                    )
                
                if success:
                    st.success("""
                    ✅ **Mensagem enviada com sucesso!**
                    
                    O professor receberá sua mensagem em instantes.
                    Aguarde o retorno pelo e-mail informado.
                    
                    Obrigado pelo seu feedback! 🎓
                    """)
                    st.balloons()
                else:
                    st.error("""
                    ❌ **Erro ao enviar mensagem**
                    
                    Ocorreu um problema no envio. Por favor, tente novamente
                    ou entre em contato diretamente por e-mail.
                    """)
    
    st.markdown("---")
    
    # FAQ
    st.subheader("❓ Perguntas Frequentes")
    
    with st.expander("Quanto tempo para receber uma resposta?"):
        st.markdown("""
        O professor receberá sua mensagem imediatamente via notificação.
        O tempo de resposta varia conforme a demanda, mas geralmente:
        
        - **Dúvidas simples:** 24-48 horas
        - **Sugestões:** Avaliadas semanalmente
        - **Erros/Bugs:** Priorizados para correção rápida
        """)
    
    with st.expander("Posso enviar anexos ou imagens?"):
        st.markdown("""
        No momento, este formulário aceita apenas texto.
        
        Se precisar enviar capturas de tela ou arquivos, mencione isso
        na mensagem e o professor entrará em contato por e-mail
        para solicitar os materiais adicionais.
        """)
    
    with st.expander("Minha dúvida é sobre um exercício específico"):
        st.markdown("""
        Para dúvidas sobre exercícios, inclua na mensagem:
        
        1. **Módulo e seção** onde está o exercício
        2. **Enunciado** ou descrição do problema
        3. **O que você tentou** fazer
        4. **Onde encontrou dificuldade**
        
        Quanto mais detalhes, melhor será o atendimento!
        """)
    
    with st.expander("Encontrei um erro no conteúdo"):
        st.markdown("""
        Obrigado por ajudar a melhorar o aplicativo!
        
        Ao reportar erros de conteúdo, informe:
        
        - **Módulo e seção** exatos
        - **O que está errado** (fórmula, explicação, etc.)
        - **O que deveria ser** (se souber)
        - **Referência** (se tiver uma fonte)
        """)
    
    
# =============================================================================
# EXECUÇÃO STANDALONE (para testes)
# =============================================================================

if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Caixa de Sugestões - Lab Econometria",
            page_icon="📬",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass
    render()