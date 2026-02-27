import streamlit as st
import requests
import sqlite3
import joblib
import numpy as np
from datetime import datetime
from textblob import TextBlob
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
# =============================
# CONFIG
# =============================



conn = sqlite3.connect("database.db", check_same_thread=False)

st.set_page_config(
    page_title="CineGuard AI",
    page_icon="🎬",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
}
.risk-box {
    padding:20px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎬 CineGuard AI</p>', unsafe_allow_html=True)
st.write("AI-Powered Movie Review Manipulation Detection")

# =============================
# FUNCTIONS
# =============================

def search_movies(query):
    url = f"http://www.omdbapi.com/?s={query}&apikey={OMDB_API_KEY}"
    response = requests.get(url).json()
    if response.get("Response") == "True":
        return response["Search"]
    return []

def fetch_movie(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}&plot=full"
    return requests.get(url).json()

def extract_features(movie):
    try:
        if movie.get("Type") != "movie":
            return None

        rating = float(movie["imdbRating"]) if movie["imdbRating"] != "N/A" else None
        votes = int(movie["imdbVotes"].replace(",", "")) if movie["imdbVotes"] != "N/A" else None
        year = int(movie["Year"].split("–")[0]) if movie["Year"] != "N/A" else None
        runtime_str = movie.get("Runtime", "N/A")
        runtime = int(runtime_str.split(" ")[0]) if runtime_str != "N/A" else None

        if None in [rating, votes, year, runtime]:
            return None

        sentiment = TextBlob(movie["Plot"]).sentiment.polarity
        movie_age = datetime.now().year - year
        ratio = rating / np.log1p(votes)

        return [rating, votes, runtime, sentiment, movie_age, ratio]

    except:
        return None

def explain_risk(score):
    if score > 60:
        return "High Risk", "🔴", """
        This movie shows abnormal rating behaviour.

        • Rating and vote pattern is unusual  
        • Possible artificial boosting or engagement manipulation  
        • Review carefully before trusting ratings  
        """
    elif score > 30:
        return "Moderate Risk", "🟡", """
        Some irregular signals detected.

        • Slightly unusual engagement pattern  
        • Not strongly manipulated but worth monitoring  
        """
    else:
        return "Low Risk", "🟢", """
        Rating behaviour appears normal.

        • Vote pattern looks organic  
        • No strong manipulation signals detected  
        """

def fraud_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Fraud Risk Score"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0,30], 'color': "#2ecc71"},
                {'range': [30,60], 'color': "#f1c40f"},
                {'range': [60,100], 'color': "#e74c3c"}
            ]
        }
    ))
    return fig

# =============================
# SEARCH UI
# =============================

query = st.text_input("Search Movie (Start typing...)")

if query:

    results = search_movies(query)

    if results:

        movie_titles = [f"{m['Title']} ({m['Year']})" for m in results]
        selected_option = st.selectbox("Select Movie", movie_titles)

        selected_title = selected_option.rsplit(" (", 1)[0]

        if st.button("Analyze Movie"):

            movie = fetch_movie(selected_title)

            if movie["Response"] == "True":

                col1, col2 = st.columns([1,2])

                with col1:
                    if movie["Poster"] != "N/A":
                        st.image(movie["Poster"], use_container_width=True)

                with col2:
                    st.subheader(movie["Title"])
                    st.write(f"**Year:** {movie['Year']}")
                    st.write(f"**Genre:** {movie['Genre']}")
                    st.write(f"**IMDb Rating:** {movie['imdbRating']}")
                    st.write(f"**IMDb Votes:** {movie['imdbVotes']}")

                features = extract_features(movie)

                if features is not None:

                    try:
                        model = joblib.load("model.pkl")
                        scaler = joblib.load("scaler.pkl")
                        min_score, max_score = joblib.load("score_range.pkl")

                        X_scaled = scaler.transform([features])
                        score = model.decision_function(X_scaled)[0]

                        # Proper normalization
                        normalized = (score - min_score) / (max_score - min_score)
                        fraud_score = round((1 - normalized) * 100, 2)

                        st.plotly_chart(fraud_gauge(fraud_score), use_container_width=True)

                        level, icon, explanation = explain_risk(fraud_score)

                        st.markdown(f"## {icon} {level}")
                        st.write(explanation)

                    except:
                        st.warning("Model not trained yet. Train model in Jupyter first.")

                else:
                    st.warning("Insufficient data for analysis.")

            else:
                st.error("Movie details not found.")

    else:
        st.warning("No matching movies found.")