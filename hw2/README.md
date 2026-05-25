# Homework 2 - Making Your Model Talk

**Μάθημα:** AI Hands-on, ΕΜΠ 

**Θεματική Ενότητα:** Ανίχνευση Προέλευσης Μουσικών Κομματιών (Spotify)

**Ονοματεπώνυμο:** Αντωνία Κατσούρη

**Αριθμός Μητρώου:** 09325010

## 1. Περιγραφή και Στόχοι
Στόχος της εργασίας είναι η δημιουργία ενός έξυπνου συνομιλιακού πράκτορα (Conversational AI Agent) που λειτουργεί ως γέφυρα μεταξύ του χρήστη και του μοντέλου Machine Learning που αναπτύχθηκε στο HW1. 

Ο πράκτορας μπορεί:
1. Να απαντά σε ερωτήσεις γνώσεων γύρω από τη μουσική, τα χαρακτηριστικά του Spotify (audio features) και την τεχνητή νοημοσύνη στη μουσική.
2. Να κάνει προβλέψεις σε πραγματικό χρόνο για το αν ένα κομμάτι είναι δημιουργημένο από άνθρωπο (Human-made) ή από Τεχνητή Νοημοσύνη (AI-Generated), αξιοποιώντας το εκπαιδευμένο Νευρωνικό Δίκτυο.
3. Να εξάγει στατιστικά στοιχεία απευθείας από το dataset.

## 2. Αρχιτεκτονική Συστήματος (High-Level Architecture)
Η αρχιτεκτονική του συστήματος βασίζεται σε σύγχρονα εργαλεία LLM orchestration και web development:
* **Web Framework:** Το API είναι στημένο με **FastAPI** (`api.py`), προσφέροντας endpoints για chat, streaming απαντήσεων και διαχείριση συνεδριών (session memory).
* **Agent Logic:** Η ροή του Agent έχει υλοποιηθεί με **LangGraph** (`agent.py`). Ο Agent δέχεται το μήνυμα του χρήστη, αξιολογεί αν χρειάζεται να χρησιμοποιήσει κάποιο εργαλείο (ToolNode) ή αν μπορεί να απαντήσει απευθείας, και τροφοδοτείται από cloud LLMs (υποστηρίζει Groq, OpenAI, Google, Anthropic).
* **Vector Store / RAG:** Χρησιμοποιείται η **ChromaDB** (`rag.py`) για την αποθήκευση και τοπική ανάκτηση γνώσης, με embeddings από το HuggingFace (`all-MiniLM-L6-v2`).
* **Machine Learning / Tools:** Το `tools.py` γεφυρώνει τον πράκτορα με το PyTorch μοντέλο του HW1.

## 3. Σύστημα RAG (Retrieval-Augmented Generation)
Για να αποκτήσει ο πράκτορας εξειδικευμένη γνώση (Domain Knowledge), συλλέχθηκαν 5 έγγραφα κειμένου που αποθηκεύτηκαν στον φάκελο `data/documents/`:
1. Get Track's Audio Features (ορισμοί των audio features) - Πηγή: https://developer.spotify.com/documentation/web-api/reference/get-audio-features
2. Artificial Intelligence in Music: Analysis, Classification, and Recommendation  - Πηγή: https://en.wikipedia.org/wiki/Artificial_intelligence_in_music
3. Music information retrieval - Πηγή: https://en.wikipedia.org/wiki/Music_information_retrieval
4. AI-Generated Music Detection and its Challenges - Πηγή: https://arxiv.org/abs/2501.10111
5. Music Genres Explained: A Sonic Guide - Πηγή: https://orphiq.com/resources/music-genres-explained

**Γιατί επιλέχθηκαν:** Τα κείμενα αυτά καλύπτουν τόσο το τεχνικό λεξιλόγιο (π.χ. τι εύρος τιμών έχει το `acousticness`) όσο και το επιστημονικό υπόβαθρο (γιατί τα AI τραγούδια ξεχωρίζουν). Το RAG κάνει chunking στα 600 characters και επιστρέφει τα κορυφαία 3 (Top-k=3) σχετικά κομμάτια.
**Τι ερωτήσεις απαντά:** *"Τι σημαίνει η μεταβλητή valence;", "Πώς τα μηχανήματα καταλαβαίνουν τα μουσικά είδη;", "Γιατί το liveness είναι σημαντικό;"*

## 4. Ενσωμάτωση Μοντέλου HW1 
Το μοντέλο που μεταφέρθηκε από το HW1 είναι το **Νευρωνικό Δίκτυο (PyTorch)** (`best_model.pt`) μαζί με τον αντίστοιχο scaler (`scaler.pkl`). 

Το **Prediction Tool** του Agent (`prediction_tool`) αναμένει ως είσοδο ένα JSON format με 15 βασικά audio features. Μόλις τα λάβει:
1. Υπολογίζει αυτόματα (Feature Engineering) τα 3 επιπλέον χαρακτηριστικά που δημιουργήσαμε στο HW1 (`energy_acousticness_ratio`, `danceability_valence_product`, `loudness_positive`).
2. Κανονικοποιεί τα 18 πλέον χαρακτηριστικά μέσω του Scikit-Learn Scaler.
3. Εκτελεί το Forward Pass στο Νευρωνικό Δίκτυο και επιστρέφει την πιθανότητα το τραγούδι να είναι Human-made ή AI-generated.

## 5. Επιπλέον Υλοποιήσεις
* **Task 5 (Dataset Analytics Tool):** Δημιουργήθηκε το `dataset_stats_tool` που επιτρέπει στον Agent να διαβάζει το CSV αρχείο και να επιστρέφει στατιστικά (μέσο όρο, min, max, κατανομές) όταν ο χρήστης ρωτάει π.χ. *"Ποιος είναι ο μέσος όρος του tempo στο dataset;"*.
* **Task 6 (Streaming API):** Υλοποιήθηκε το endpoint `POST /chat/stream` το οποίο επιστρέφει την απάντηση σταδιακά (token-by-token) χρησιμοποιώντας το πρωτόκολλο Server-Sent Events (SSE).

## 6. Παραδείγματα συζητήσεων
## Παράδειγμα 1

## Παράδειγμα 2
Χρήστης: Is this track human or AI? acousticness=0.05, danceability=0.85, duration_ms=210000, energy=0.91, instrumentalness=0.82, key=5, liveness=0.08, loudness=-4.5, mode=1, speechiness=0.03, tempo=128.0, time_signature=4, valence=0.6, popularity=40, short_form=0
Απάντηση: The provided track is: Human-made.
Χρήστης: Why did you classify this track as Human-made? Which audio features played the biggest role in this decision?"
Απάντηση: The audio features that played the biggest role in classifying the track as Human-made include:\n\n1. **Instrumentalness**: The track has a high instrumentalness value (0.82), which suggests that it features a lot of instrumental elements, such as guitars, drums, or synthesizers. Human-made music often has a more nuanced and varied instrumental arrangement, whereas AI-generated music may rely more heavily on pre-programmed loops or patterns.\n2. **Liveness**: The track has a low liveness value (0.08), which indicates that it was likely recorded in a studio or produced using digital tools. Human-made music often has a more dynamic and expressive quality, with subtle variations in timing, tone, and texture that are harder to replicate with AI-generated music.\n3. **Speechiness**: The track has a very low speechiness value (0.03), which suggests that it does not feature any significant spoken word elements. Human-made music often incorporates spoken word or vocal elements in a more creative and expressive way, whereas AI-generated music may struggle to replicate the nuances of human speech.\n4. **Valence**: The track has a moderate valence value (0.6), which indicates that it has a somewhat positive or uplifting emotional tone. Human-made music often has a more complex and nuanced emotional landscape, with a wider range of emotional expression and depth.\n\nThese audio features, among others, contributed to the classification of the track as Human-made. However, it's worth noting that the distinction between Human-made and AI-generated music can be subtle, and the classification may not always be accurate.


## 7. Εγκατάσταση και Εκτέλεση (Installation & Execution)

**Βήμα 1: Κλωνοποίηση του repository**
```bash
git clone https://github.com/antoniakatsouri1-math/Spotify-Ai-Classifier.git
cd Spotify-Ai-Classifier/hw2
```
**Βήμα 2: Δημιουργία Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate
```

**Βήμα 3: Εγκαθιστούμε τις βιβλιοθήκες**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**Βήμα 4: Ρύθμιση Μεταβλητών Περιβάλλοντος**
Δημιουργήστε ένα αρχείο .env στον φάκελο hw2 και προσθέστε το API Key σας. Το σύστημα έχει ως προεπιλογή το Groq (Llama 3), αλλά υποστηρίζει και OpenAI, Google ή Anthropic.
```bash
LLM_PROVIDER=groq GROQ_API_KEY=ΤΟ_GROQ_API_KEY_ΣΑΣ python3 main.py
```
**Βήμα 5: Εκκίνηση του FastAPI Server**
```bash
python main.py
```
(Ο server θα ξεκινήσει στο http://0.0.0.0:8000. Το vector store θα δημιουργηθεί αυτόματα την πρώτη φορά που θα τρέξετε τον κώδικα).

**Βήμα 6: ΠεριβάλλονSwagger UI**
Επισκεφθείτε το http://127.0.0.1:8000/docs 


