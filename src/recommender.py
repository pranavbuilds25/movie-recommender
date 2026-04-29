import os
import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# -------------------------
# LOAD ENV
# -------------------------
load_dotenv()

TMDB_KEY = os.getenv("TMDB_API_KEY")
OMDB_KEY = os.getenv("OMDB_API_KEY")

if not TMDB_KEY:
    raise ValueError("TMDB_API_KEY missing in .env")

# -------------------------
# LOAD CACHE (LOCAL SPEED BOOST)
# -------------------------
try:
    with open("poster_cache.json", "r") as f:
        poster_cache = json.load(f)
except:
    poster_cache = {}


# -------------------------
# LOAD DATA
# -------------------------
def load_data():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")
    return movies.merge(credits, on="title")


# -------------------------
# PREPROCESS
# -------------------------
def preprocess(movies):

    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords']]
    movies.dropna(inplace=True)

    movies['tags'] = (
        movies['overview'] + " " +
        movies['genres'] + " " +
        movies['keywords']
    ).str.lower()

    return movies


# -------------------------
# SIMILARITY
# -------------------------
def compute_similarity(movies):

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags'])

    return cosine_similarity(vectors)


# -------------------------
# HYBRID POSTER SYSTEM
# -------------------------
def fetch_poster(movie_id, title=None):

    movie_id = str(movie_id)

    # 1️⃣ CACHE (FASTEST)
    if movie_id in poster_cache:
        return poster_cache[movie_id]

    # 2️⃣ TMDB API
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_KEY}"
        r = requests.get(url, timeout=4)

        if r.status_code == 200:
            data = r.json()
            poster_path = data.get("poster_path")

            if poster_path:
                url = "https://image.tmdb.org/t/p/w500" + poster_path
                poster_cache[movie_id] = url
                return url

    except:
        pass

    # 3️⃣ OMDb API (FALLBACK)
    if title and OMDB_KEY:
        try:
            url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_KEY}"
            data = requests.get(url, timeout=4).json()

            poster = data.get("Poster")

            if poster and poster != "N/A":
                poster_cache[movie_id] = poster
                return poster

        except:
            pass

    # 4️⃣ FINAL FALLBACK (NEVER FAIL UI)
    fallback = "https://via.placeholder.com/300x450?text=No+Poster"
    poster_cache[movie_id] = fallback
    return fallback


# -------------------------
# RECOMMENDATION ENGINE
# -------------------------
def recommend(movie_name, movies, similarity):

    if movie_name not in movies['title'].values:
        return [], []

    idx = movies[movies['title'] == movie_name].index[0]

    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    names = []
    posters = []

    for i in distances:
        row = movies.iloc[i[0]]

        names.append(row.title)
        posters.append(fetch_poster(row.movie_id, row.title))

    return names, posters


# -------------------------
# SAVE CACHE (OPTIONAL BUT RECOMMENDED)
# -------------------------
def save_cache():
    with open("poster_cache.json", "w") as f:
        json.dump(poster_cache, f)
