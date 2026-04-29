import streamlit as st
from src.recommender import load_data, preprocess, compute_similarity, recommend, save_cache

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Netflix Style Movie Recommender",
    layout="wide"
)

# -------------------------
# CUSTOM UI STYLE (NETFLIX DARK THEME)
# -------------------------
st.markdown("""
    <style>

    body {
        background-color: #0e1117;
        color: white;
    }

    .movie-title {
        text-align: center;
        font-size: 14px;
        margin-top: 5px;
        color: #ffffff;
    }

    .stButton button {
        background-color: #e50914;
        color: white;
        border-radius: 8px;
        height: 45px;
        width: 100%;
        font-weight: bold;
    }

    .stButton button:hover {
        background-color: #b00610;
    }

    </style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.title("🎬 Netflix Style Movie Recommender")

st.markdown("---")

# -------------------------
# CACHE DATA
# -------------------------
@st.cache_data
def get_data():
    movies = load_data()
    movies = preprocess(movies)
    similarity = compute_similarity(movies)
    return movies, similarity


movies, similarity = get_data()

# -------------------------
# MOVIE SELECT
# -------------------------
selected_movie = st.selectbox("🔍 Search Movie", movies['title'].values)

# -------------------------
# RECOMMEND BUTTON
# -------------------------
if st.button("Recommend"):

    names, posters = recommend(selected_movie, movies, similarity)

    st.markdown("## 🍿 Recommended for you")

    # -------------------------
    # DISPLAY CARDS
    # -------------------------
    cols = st.columns(5)

    for i in range(len(names)):

        with cols[i]:

            if posters[i]:
                st.image(posters[i], use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x450?text=No+Image",
                         use_container_width=True)

            st.markdown(
                f"<div class='movie-title'>{names[i]}</div>",
                unsafe_allow_html=True
            )

# -------------------------
# SAVE CACHE (OPTIONAL)
# -------------------------
try:
    save_cache()
except:
    pass
