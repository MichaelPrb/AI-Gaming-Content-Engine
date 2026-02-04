"""
recommender_engine.py

Module: Game Recommendation System
Objective: 
Builds a content-based filtering system using TF-IDF and Cosine Similarity
based on the enriched metadata from Module 2.

Author: Michael Emmanuel Purba
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Configuration ---
# Input must be the OUTPUT from Module 2
INPUT_DATA = "data/enriched_games.csv"

def load_and_prep_data(filepath):
    """Loads data and creates a combined feature string for analysis."""
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded {len(df)} games for recommendation engine.")
        
        # Fill NaN values to avoid errors
        df = df.fillna('')
        
        # Create a "soup" of metadata (combining all features into one string)
        df['combined_features'] = (
            df['genre'] + " " + 
            df['short_description'] + " " + 
            df['player_mode']
        )
        return df
    except FileNotFoundError:
        print(f"[FATAL] File not found: {filepath}. Run 'game_enrichment.py' first!")
        exit()

def build_similarity_matrix(df):
    """Calculates Cosine Similarity matrix using TF-IDF."""
    print("[INFO] Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Convert text to matrix
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Calculate cosine similarity
    print("[INFO] Calculating cosine similarity...")
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, df, cosine_sim):
    """Returns top 5 similar games based on the title."""
    try:
        # Get the index of the game that matches the title
        indices = pd.Series(df.index, index=df['game_title']).drop_duplicates()
        
        if title not in indices:
            return [f"Game '{title}' not found in database."]

        idx = indices[title]

        # Get the pairwise similarity scores of all games with that game
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the games based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar games
        sim_scores = sim_scores[1:6]

        # Get the game indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar games
        return df['game_title'].iloc[movie_indices].tolist()
    
    except Exception as e:
        return [f"Error processing recommendation: {e}"]

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Load Data
    df_games = load_and_prep_data(INPUT_DATA)
    
    # 2. Build Engine
    similarity_matrix = build_similarity_matrix(df_games)
    
    # 3. Test Cases (Interactive or Static)
    test_games = ["Valorant", "Elden Ring", "Minecraft"]
    
    print("\n--- Recommendation Results ---")
    for game in test_games:
        print(f"\nTarget Game: {game}")
        recs = get_recommendations(game, df_games, similarity_matrix)
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec}")