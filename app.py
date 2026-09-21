import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from agente import interroga_agente

# ==========================================
# Configurazione Layout Streamlit
# ==========================================
st.set_page_config(page_title="GraphRAG UNIMORE", page_icon="🎓", layout="centered")

st.title("🎓 Assistente GraphRAG - UNIMORE")
st.markdown("Interroga il Knowledge Graph accademico (Neo4j) in linguaggio naturale.")

# Gestione dello stato della sessione per la memoria della chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Rendering dello storico dei messaggi
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Gestione del nuovo input utente
if prompt := st.chat_input("Cerca paper, autori o dipartimenti..."):
    
    # Rendering input utente
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Elaborazione e rendering risposta agente
    with st.chat_message("assistant"):
        # Callback handler per visualizzare in tempo reale i passaggi di ragionamento dell'agente (Explainable AI)
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        
        with st.spinner("Estrazione dati dal Knowledge Graph in corso..."):
            try:
                risposta = interroga_agente(st.session_state.messages, callbacks=[st_callback])
                st.markdown(risposta)
                st.session_state.messages.append({"role": "assistant", "content": risposta})
                
            except Exception as e:
                st.error(f"Si è verificato un errore di elaborazione: {e}")