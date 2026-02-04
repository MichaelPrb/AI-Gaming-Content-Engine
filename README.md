# 🎮 AI Gaming Content Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![GenAI](https://img.shields.io/badge/AI-Google%20Gemini-orange)
![Sklearn](https://img.shields.io/badge/ML-Scikit--Learn-green)

An end-to-end AI system designed to understand gaming contexts, enrich game metadata using LLMs, and provide content-based game recommendations.

## 🚀 Project Modules

### Module 1: Game Event Detection Logic (System Design)
Designed the detection logic for automated gaming highlights (e.g., "Ace", "Clutch", "Victory") using Audio/Visual triggers.
* **Documentation:** [View Logic Breakdown](docs/game_event_logic.pptx) (Located in `docs/` folder)
* **Key Logic:** Defined Pre-roll/Post-roll durations and specific UI/Audio triggers for Valorant.

### Module 2: Metadata Enrichment (GenAI)
Automated the enrichment of raw game lists with metadata (Genre, Description, Player Mode) using **Google Gemini 2.0 Flash**.
* **Script:** `modules/game_enrichment.py`
* **Tech:** LangChain, Google Generative AI, Pandas.
* **Output:** Converts raw game titles into structured datasets.

### Module 3: Recommendation Engine (ML)
Implements a Content-Based Filtering system to recommend games based on the enriched metadata from Module 2.
* **Script:** `modules/recommender_engine.py`
* **Tech:** TF-IDF Vectorizer, Cosine Similarity.
* **Features:** Recommends top 5 similar games based on genre and description analysis.

## 📂 Repository Structure

```text
├── data/
│   ├── raw_games.csv           # Initial input data
│   └── enriched_games.csv      # AI-processed data (Result of Module 2)
├── modules/c.
│   ├── game_enrichment.py      # Script for GenAI Metadata processing
│   └── recommender_engine.py   # Script for ML Recommendations
├── .env                        # API Keys (Not uploaded)
└── requirements.txt            # Dependencies
