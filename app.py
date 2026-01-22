# app.py (обновлённая версия)
import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from gigachat_rag import generate_rag_response  # ← наша новая функция

@st.cache_resource
def load_search():
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    index = faiss.read_index("data/embedding_model/movie_index.faiss")
    with open("data/embedding_model/movies.pkl", "rb") as f:
        movies = pickle.load(f)
    return model, index, movies

model, index, all_movies = load_search()

st.title("🎥 Семантический поиск + RAG")
query = st.text_input("Опишите фильм:", placeholder="Пример: фильм про ведьму в замке")

if query:
    with st.spinner("Ищем фильмы..."):
        # Поиск
        emb = model.encode("query: " + query, normalize_embeddings=True)
        D, I = index.search(np.array([emb]).astype('float32'), k=3)
        
        top_movies = [all_movies[idx] for idx in I[0]]
        
        # RAG-ответ
        with st.spinner("Генерируем рекомендацию..."):
            rag_text = generate_rag_response(query, top_movies)
        
        # Вывод
        st.subheader("🧠 Рекомендация от ИИ")
        st.write(rag_text)
        
        st.divider()
        st.subheader("🎬 Найденные фильмы")
        
        for m in top_movies:
            col1, col2 = st.columns([1, 4])
            with col1:
                poster = m.get("poster_url", "")
                if poster and "placeholder" not in poster:
                    st.image(poster, width=120)
                else:
                    st.write("🖼️ Нет постера")
            with col2:
                year = int(m["year"]) if pd.notna(m["year"]) else "?"
                st.subheader(f"{m['title']} ({year})")
                st.write(m["description"][:300] + "...")
                st.markdown(f"[Подробнее]({m['tmdb_url']})", unsafe_allow_html=True)
            st.divider()