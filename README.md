# 🎬 Movie Recommender System

## 📌 Overview
A content-based movie recommendation system that suggests similar movies based on genres using cosine similarity.

## 🚀 Features
- Select a movie from dropdown
- Get top 5 similar movie recommendations
- Interactive UI built with Streamlit

## 🧠 How it works
- Uses movie genres as features
- Converts text to vectors using CountVectorizer
- Computes similarity using cosine similarity
- Recommends top similar movies

## 🛠 Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
