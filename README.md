# 🎵 Spotify AI Track Classifier

**Μάθημα:** AI Hands-on, ΕΜΠ  
**Φοιτήτρια:** Αντωνία Κατσούρη | ΑΜ: 09325010  
**Repository:** [github.com/antoniakatsouri1-math/Spotify-Ai-Classifier](https://github.com/antoniakatsouri1-math/Spotify-Ai-Classifier)

---

## Περιγραφή Έργου

Αυτό το project αναπτύχθηκε σε δύο φάσεις και στοχεύει στην **αυτόματη ανίχνευση προέλευσης μουσικών κομματιών** — διακρίνοντας αν ένα τραγούδι στο Spotify έχει δημιουργηθεί από άνθρωπο ή από Τεχνητή Νοημοσύνη, βάσει των ακουστικών χαρακτηριστικών του.

---

## HW1 — From Data to Intelligent Model

### Πρόβλημα

Ταξινόμηση μουσικών κομματιών ως **Human-made** ή **AI-Generated** με βάση audio features (acousticness, energy, tempo, κ.ά.). Η ανίχνευση αυτή έχει άμεση αξία για πλατφόρμες streaming, οργανισμούς διαχείρισης δικαιωμάτων και καλλιτέχνες.

### Dataset

| Στοιχείο | Τιμή |
|---|---|
| Πηγή | Spotify 2026 Synthetic Dataset — Kaggle |
| Μέγεθος | 391.989 γραμμές × 20 στήλες |
| Κλάση 0 (Human) | 261.326 δείγματα |
| Κλάση 1 (AI-Generated) | 130.663 δείγματα |

### Preprocessing Pipeline

- **Split:** Stratified 80/10/10 (train/val/test), `random_state=42`
- **Missing Values:** Imputation με διάμεσο (median) του training set
- **Outliers:** IQR Winsorizing — τα όρια υπολογίστηκαν μόνο στο train set
- **Scaling:** `StandardScaler` — αποθηκευμένος ως `models/scaler.pkl`
- **Αφαίρεση στηλών:** `artist_name`, `track_id`, `track_name`, `scenario` (υψηλή cardinality)
- **Ανισορροπία κλάσεων:** `class_weight='balanced'` (RF) & weighted loss (NN)

#### Feature Engineering

| Feature | Υπολογισμός | Σκεπτικό |
|---|---|---|
| `energy_acousticness_ratio` | `energy / (acousticness + 1e-6)` | AI κομμάτια: υψηλή energy + χαμηλή acousticness (r = −0.71) |
| `danceability_valence_product` | `danceability × valence` | AI συστήματα βελτιστοποιούν ταυτόχρονα τους δύο άξονες |
| `loudness_positive` | `loudness + 60.0` | Μετατόπιση σε μη-αρνητική κλίμακα για PCA |

### Αποτελέσματα Μοντέλων

| Metric | Random Forest | Neural Network |
|---|---|---|
| Accuracy | 0.885 | **0.898** |
| Precision | 0.786 | **0.862** |
| Recall | **0.898** | 0.825 |
| F1-Score | 0.838 | **0.843** |
| ROC-AUC | 0.958 | **0.961** |

**Επιλεγμένο μοντέλο:** Neural Network (PyTorch)  
Αρχιτεκτονική: `Input(18) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, LeakyReLU) → Dense(1, Sigmoid)`

### API (FastAPI)

Εκτελέστε:
```bash
cd hw1/src
python3 -m uvicorn api:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`

**Endpoint:** `POST /predict`

```json
// Input
{
  "acousticness": 0.35,
  "danceability": 0.68,
  "duration_ms": 210000,
  "energy": 0.72,
  "instrumentalness": 0.05,
  "key": 5,
  "liveness": 0.12,
  "loudness": -8.5,
  "mode": 1,
  "speechiness": 0.04,
  "tempo": 122.0,
  "time_signature": 4,
  "valence": 0.55,
  "popularity": 40,
  "short_form": 0
}

// Output
{
  "prediction": 0,
  "label": "Human",
  "probability": 0.23
}
```

### Εγκατάσταση HW1

```bash
git clone https://github.com/antoniakatsouri1-math/Spotify-Ai-Classifier
cd hw1
pip install -r requirements.txt
python main.py        # Preprocessing + Training + Evaluation
python api.py         # REST API
```

---

## HW2 — Making Your Model Talk

### Στόχος

Δημιουργία ενός **Conversational AI Agent** που χρησιμεύει ως γέφυρα μεταξύ χρήστη και του ML μοντέλου του HW1. Ο agent μπορεί να απαντά σε ερωτήσεις γνώσης, να κάνει προβλέψεις σε πραγματικό χρόνο και να εξάγει στατιστικά από το dataset.

### Αρχιτεκτονική

```
[User Request] ──> [FastAPI (api.py)] ──> [LangGraph Engine (agent.py)]
                                                    │
                 ┌──────────────────────────────────┼──────────────────────────────┐
                 ▼                                  ▼                              ▼
       [rag_knowledge_tool]               [prediction_tool]            [dataset_stats_tool]
         (ChromaDB Vector)                (PyTorch NN Model)               (Pandas CSV)
```

| Στοιχείο | Τεχνολογία |
|---|---|
| Web Framework | FastAPI |
| Agent Logic | LangGraph |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM Support | Groq / OpenAI / Google / Anthropic |
| ML Model | PyTorch Neural Network (από HW1) |

### Σύστημα RAG

Ο agent αξιοποιεί 5 έγγραφα γνώσης (chunk size: 600 χαρακτήρες, Top-k: 3):

1. **Spotify Audio Features API** — Τεχνικοί ορισμοί και όρια χαρακτηριστικών
2. **AI in Music (Wikipedia)** — Ιστορικό & θεωρητικό πλαίσιο ML + ήχος
3. **Music Information Retrieval (Wikipedia)** — Επιστημονικές μέθοδοι ανάλυσης ήχου
4. **AI-Generated Music Detection (arXiv)** — Σύγχρονη έρευνα ανίχνευσης AI μουσικής
5. **Music Genres Explained** — Ποιοτικά χαρακτηριστικά μουσικών ειδών

### Prediction Tool

Το tool δέχεται 15 audio features και αυτόματα:
1. Υπολογίζει τα 3 engineered features (από HW1)
2. Κανονικοποιεί με το αποθηκευμένο `scaler.pkl`
3. Εκτελεί forward pass στο Neural Network
4. Επιστρέφει label + confidence probability

### Παραδείγματα Συζητήσεων

**Ερώτηση γνώσης:**
```
User:  What audio features make a song popular on Spotify?
Agent: Danceability, energy, tempo, and acousticness are key features...
```

**Στατιστικά dataset:**
```
User:  What is the average tempo in the dataset?
Agent: The average tempo in the dataset is 119.47 beats per minute.
```

**Πρόβλεψη κομματιού:**
```
User:  Is this track human or AI?
       acousticness=0.05, energy=0.91, instrumentalness=0.82, tempo=128.0 ...

Agent: The track is predicted to be HUMAN-MADE with 73.5% confidence.
       The low acousticness and high instrumentalness are consistent with
       human-made electronic music patterns.
```

### Εγκατάσταση HW2

```bash
# Βήμα 1: Clone & navigate
git clone https://github.com/antoniakatsouri1-math/Spotify-Ai-Classifier.git
cd Spotify-Ai-Classifier/hw2

# Βήμα 2: Virtual environment
python3 -m venv venv
source venv/bin/activate

# Βήμα 3: Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Βήμα 4: Ρύθμιση API key (.env αρχείο)
LLM_PROVIDER=groq GROQ_API_KEY=ΤΟ_GROQ_API_KEY_ΣΑΣ python3 main.py
```

Swagger UI: `http://127.0.0.1:8000/docs`

**Παράδειγμα κλήσης:**
```bash
curl -X POST 'http://127.0.0.1:8000/chat' \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is the role of acousticness in music classification?", "session_id": "session_1"}'
```

**Streaming endpoint:** `POST /chat/stream` (Server-Sent Events, token-by-token)

---

## Δομή Repository

```
Spotify-Ai-Classifier/
├── hw1/
│   ├── src/
│   │   ├── preprocessing.py
│   │   ├── api.py
│   │   └── ...
│   ├── models/
│   │   ├── best_model.pt
│   │   └── scaler.pkl
│   ├── main.py
│   └── requirements.txt
└── hw2/
    ├── src/
    │   ├── agent.py
    │   ├── tools.py
    │   ├── rag.py
    │   └── api.py
    ├── data/documents/
    ├── main.py
    └── requirements.txt
```

---

## Τεχνολογίες

`Python` · `PyTorch` · `Scikit-learn` · `FastAPI` · `LangGraph` · `ChromaDB` · `HuggingFace` · `Pandas` · `Groq API`
