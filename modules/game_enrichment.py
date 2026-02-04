"""
game_enrichment.py

Module: Gaming Metadata Enrichment
Objective: 
Loads a raw CSV of game titles and uses Generative AI (Google Gemini) 
to enrich it with Genre, Short Description, and Player Mode metadata.

Author: Michael Emmanuel Purba
"""

import pandas as pd
import os
import time
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Path Configuration
# Assumes script is run from the root directory
INPUT_CSV = "data/raw_games.csv"
OUTPUT_CSV = "data/enriched_games.csv"
GAME_COLUMN_NAME = "game_title" 

MODEL_NAME = "gemini-2.0-flash" 
SLEEP_TIME_PER_CALL = 5 

if not API_KEY:
    print("[FATAL] GOOGLE_API_KEY not found in .env file.")
    exit()

print("[INFO] Initializing AI Engine...")

# --- AI Setup ---
model = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=API_KEY, max_retries=1)

prompt_template = PromptTemplate.from_template(
    """
    Anda adalah asisten data game profesional. 
    Berdasarkan nama game, berikan tiga informasi berikut:
    1. genre: SATU kata (Contoh: 'Shooter', 'RPG', 'Strategy')
    2. short_description: Penjelasan singkat di BAWAH 30 kata.
    3. player_mode: HANYA 'Singleplayer', 'Multiplayer', atau 'Both'.

    Format jawaban HARUS JSON valid.
    Contoh:
    {{
      "genre": "Shooter",
      "short_description": "Tactical 5v5 shooter game.",
      "player_mode": "Multiplayer"
    }}  

    Game: {nama_game}
    JSON:
    """
)

chain = prompt_template | model | StrOutputParser()

# --- Main Logic ---
try:
    print(f"[INFO] Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Initialize columns if not exist
    for col in ['genre', 'short_description', 'player_mode']:
        if col not in df.columns: df[col] = ""

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="AI Processing"):
        game_title = row[GAME_COLUMN_NAME]
        
        # Skip if already processed
        if df.at[index, 'genre'] != "" and pd.notna(df.at[index, 'genre']):
            continue

        try:
            # AI Inference
            response_str = chain.invoke({"nama_game": game_title})
            
            # Clean JSON markdown if present
            if "```json" in response_str:
                response_str = response_str.split("```json\n")[1].split("```")[0]
            
            data = json.loads(response_str)

            df.at[index, 'genre'] = data.get('genre', 'N/A')
            df.at[index, 'short_description'] = data.get('short_description', 'N/A')
            df.at[index, 'player_mode'] = data.get('player_mode', 'N/A')
            
            time.sleep(SLEEP_TIME_PER_CALL) 

        except Exception as e:
            print(f"[WARN] Error on '{game_title}': {e}")
            df.at[index, 'genre'] = "ERROR"
            time.sleep(SLEEP_TIME_PER_CALL)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[SUCCESS] Data saved to {OUTPUT_CSV}")

except FileNotFoundError:
    print(f"[FATAL] File not found: {INPUT_CSV}. Please check 'data' folder.")
except Exception as e:
    print(f"[FATAL] Unexpected error: {e}")