import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def load_data():
    movies = pd.read_csv("data/movies.csv")
    return movies

def preprocess(movies):
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    return movies

def compute_similarity(movies):
    cv = CountVectorizer()
    vectors = cv.fit_transform(movies['genres'])   # NO .toarray()
    similarity = cosine_similarity(vectors)
    return similarity

def recommend(movie_name, movies, similarity):
    if movie_name not in movies['title'].values:
        return ["Movie not found"]
    
    idx = movies[movies['title'] == movie_name].index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    
    return [movies.iloc[i[0]].title for i in distances]
