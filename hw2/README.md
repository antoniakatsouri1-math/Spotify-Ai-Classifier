# HW2 – Spotify Music Intelligence Agent

**Hands-on AI · SEMFE, NTUA**

A domain-aware conversational AI agent built on top of the Spotify track popularity classifier from HW1. The agent can answer natural-language questions about music and audio features (via RAG), predict whether a track will be popular (via the HW1 model), and report dataset statistics on demand — all through a FastAPI REST interface.

---

## 1. System Overview

**Domain:** Spotify music — audio features, genres, and track popularity.

**Task carried from HW1:** Binary classification — predicting whether a track is *Popular* (popularity score ≥ 50) or *Not Popular* based on its audio features (danceability, energy, loudness, valence, tempo, etc.).

**What the agent can do:**
- Answer factual questions about Spotify audio features, genres, streaming economics, and music ML (via RAG over 5 curated domain documents)
- Predict track popularity from audio features using the HW1 best model
- Return summary statistics for any column in the Spotify dataset
- Maintain multi-turn conversation memory per session

---

## 2. Architecture

### LangGraph Graph Structure

```
START → llm_node → (tool call?) → tool_node → llm_node → END
                        ↓ no
                       END
```

The LLM is bound to all three tools. At each turn, it autonomously decides whether to call a tool (and which one) based on the user's message. Tool results are fed back to the LLM, which synthesizes a final natural-language response.

### Tools

| Tool | When Used | Source |
|------|-----------|--------|
| `retrieval_tool` | Factual/conceptual questions about music, Spotify features, genres | `src/rag.py` + ChromaDB |
| `prediction_tool` | User provides audio features and wants a popularity prediction | `models/best_model.pkl` + `models/scaler.pkl` |
| `dataset_stats_tool` *(Bonus)* | User asks for statistics on any dataset column | `data/dataset.csv` |

### Tool Selection Logic

The LLM uses its system prompt and the tool descriptions to decide autonomously:
- Questions like *"What makes a song popular?"* → `retrieval_tool`
- Input like *"danceability=0.8, energy=0.7..."* → `prediction_tool`
- Questions like *"What is the average tempo?"* → `dataset_stats_tool`
- Follow-up questions reference prior context via the maintained message history

---

## 3. Knowledge Base

**Five documents collected** (stored in `data/documents/`):

| File | Contents | Questions It Answers |
|------|----------|----------------------|
| `spotify_audio_features.txt` | Detailed description of all Spotify audio features (danceability, energy, valence, etc.) | "What does valence mean?", "How is instrumentalness measured?" |
| `music_popularity_factors.txt` | Factors driving popularity: playlist placement, recency, danceability profiles, genre differences | "Why do some songs become popular?", "Does danceability affect popularity?" |
| `music_genres_overview.txt` | Audio feature profiles for 9 major genres (pop, hip-hop, rock, EDM, jazz, classical, etc.) | "What makes EDM different from classical?", "What tempo is typical for hip-hop?" |
| `music_ml_recommendation.txt` | Spotify's recommendation system, ML approaches to popularity prediction, evaluation metrics | "How does Spotify's algorithm work?", "What ML models are used for music classification?" |
| `spotify_platform_overview.txt` | Spotify platform facts, streaming economics, discovery mechanisms, global trends | "How many tracks are on Spotify?", "How does Spotify pay artists?" |

**Chunking:** `RecursiveCharacterTextSplitter` with `chunk_size=600`, `chunk_overlap=100`.
**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API key needed).
**Vector store:** ChromaDB, persisted to `data/vector_store/`.

---

## 4. HW1 Model Integration

**Model carried forward:** RandomForestClassifier (HW1 best model based on test set ROC-AUC).

**What the prediction tool does:**
1. Accepts a JSON string with 14 audio feature fields
2. Applies the same feature engineering as HW1 (`energy_danceability_product`, `loudness_normalized`)
3. Encodes the `explicit` boolean as int
4. Scales all features using the saved `scaler.pkl`
5. Calls `model.predict()` and `model.predict_proba()`
6. Returns a human-readable prediction + probability string

**Required input fields:**
`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `explicit`, `key`, `mode`, `time_signature`

---

## 5. Example Conversations

### Example A – RAG Retrieval

```
User:   What audio features are most associated with popular tracks on Spotify?

Agent:  Based on the knowledge base, popular tracks on Spotify tend to score
        high on danceability (>0.7) and energy (>0.6), with moderate-to-high
        valence. These features align well with mainstream playlist aesthetics —
        playlists like "Today's Top Hits" and "RapCaviar" favour energetic,
        danceable tracks. Loudness is also a predictor: louder tracks (closer to
        0 dB) tend to perform better in pop and hip-hop genres. Acousticness and
        instrumentalness are inversely related to popularity in mainstream genres.
```

### Example B – Popularity Prediction

```
User:   Would a track with danceability=0.85, energy=0.72, loudness=-4.5,
        speechiness=0.06, acousticness=0.05, instrumentalness=0.0, liveness=0.12,
        valence=0.65, tempo=125, duration_ms=210000, explicit=0, key=5,
        mode=1, time_signature=4 be popular?

Agent:  [calls prediction_tool]
        Prediction: POPULAR (probability of being popular: 78.4%)
        This track is predicted to be popular based on its audio features.
        Key drivers likely include its high danceability (0.85) and energy (0.72),
        which align well with what the literature identifies as the primary
        predictors of chart success on streaming platforms.

User:   What if the acousticness was 0.85 instead?

Agent:  [calls prediction_tool with updated acousticness]
        Prediction: NOT POPULAR (probability of being popular: 23.1%)
        With a much higher acousticness of 0.85, the track is now predicted to be
        in the lower popularity tier. This is consistent with the knowledge base —
        highly acoustic tracks tend to have narrower, genre-specific audiences
        rather than mainstream chart appeal.
```

---

## 6. Installation & Execution

### Prerequisites
- Python 3.10+
- Your HW1 `best_model.pkl` and `scaler.pkl`
- An API key for OpenAI or Anthropic

### Steps

```bash
# 1. Clone the repo (or extend your HW1 repo)
git clone <your-repo-url>
cd hw2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy your HW1 artifacts
cp ../hw1/models/best_model.pkl models/
cp ../hw1/models/scaler.pkl models/
cp ../hw1/data/dataset.csv data/

# 4. Create your .env file (never commit this file!)
cp .env.example .env
# Then edit .env and add your API key

# 5. Start the server
python main.py
# The vector store is built automatically on first run and loaded from disk
# on subsequent runs. Visit http://127.0.0.1:8000/docs to explore the API.
```

### .env file format

```env
# Choose one provider
LLM_PROVIDER=openai        # or: anthropic

# Set the corresponding key
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

---

## 7. Example API Call

### Using curl

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What makes a track popular on Spotify?", "session_id": "user_001"}'
```

### Using Python requests

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "message": "Would a track with danceability=0.9 and energy=0.8 be popular?",
        "session_id": "user_001"
    }
)
print(response.json())
# {"response": "...", "session_id": "user_001"}
```

### Streaming endpoint (Bonus)

```python
import requests

with requests.post(
    "http://127.0.0.1:8000/chat/stream",
    json={"message": "Explain danceability.", "session_id": "stream_test"},
    stream=True
) as r:
    for line in r.iter_lines():
        if line and line != b"data: [DONE]":
            data = line.decode().removeprefix("data: ")
            import json
            token = json.loads(data)["token"]
            print(token, end="", flush=True)
```

---

## Bonus Features

- **Task 5 – Third Tool (`dataset_stats_tool`):** Returns descriptive statistics (mean, std, quartiles for numeric columns; value counts for categorical columns) for any column in the Spotify dataset. Invoked when users ask about data distributions.
- **Task 6 – Streaming (`/chat/stream`):** SSE streaming endpoint that delivers the agent's response word-by-word. Separate from `/chat` so both remain available.
