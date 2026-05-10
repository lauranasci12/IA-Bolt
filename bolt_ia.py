import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Interface Visual (Mantendo o estilo Neon que você gosta)
st.set_page_config(page_title="IA Bolt", page_icon="⚡")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #00FF00; }
    .bolt-circle { 
        width: 100px; height: 100px; border-radius: 50%; 
        background: #00FF00; box-shadow: 0 0 40px #00FF00; 
        margin: 20px auto; 
    }
    </style>
    <div class="bolt-circle"></div>
    """, unsafe_allow_html=True)

st.title("⚡ BOLT: IA Tech & Business")

# Função para carregar e treinar
def treinar_bolt():
    df = pd.read_csv('conhecimento.txt', sep=';', names=['texto', 'label'])
    vetorizador = TfidfVectorizer()
    X = vetorizador.fit_transform(df['texto'])
    modelo = MultinomialNB()
    modelo.fit(X, df['label'])
    return vetorizador, modelo, df

vetor, modelo, dados = treinar_bolt()

# Chatbot
pergunta = st.text_input("Pergunte ao Bolt sobre Negócios ou Tech:")

if pergunta:
    predicao = modelo.predict(vetor.transform([pergunta]))[0]
    
    if predicao == "empreendedorismo":
        st.write("🤖 **Bolt:** Isso soa como uma estratégia de **Negócios**!")
    else:
        st.write("🤖 **Bolt:** Isso parece ser sobre **Tecnologia**!")

    # A PARTE DA EVOLUÇÃO
    st.write("---")
    feedback = st.radio("Eu acertei?", ("Sim", "Não"))
    
    if feedback == "Não":
        correto = st.selectbox("Qual era a categoria correta?", ("empreendedorismo", "tecnologia"))
        if st.button("Ensinar ao Bolt"):
            with open('conhecimento.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n{pergunta};{correto}")
            st.success("Obrigado! Aprendi algo novo. Na próxima vez estarei mais inteligente.")