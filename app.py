import streamlit as st
from src.recommender import load_data, preprocess, compute_similarity, recommend

st.title("🎬 Movie Recommender System")

st.write("Step 1: Starting app...")

movies = load_data()
st.write("Step 2: Data loaded", movies.shape)

movies = preprocess(movies)
st.write("Step 3: Data preprocessed")

similarity = compute_similarity(movies)
st.write("Step 4: Similarity computed")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):
    results = recommend(selected_movie, movies, similarity)
    st.subheader("Recommended Movies:")
    for movie in results:
        st.write(movie)
