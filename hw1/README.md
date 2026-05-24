# Homework 1 — From Data to Intelligent Model
**Hands-on AI · SEMFE, NTUA**

---

## 1. Problem Description

**Domain:** Student digital behaviour and social media impact on academic/cognitive outcomes.

**Task type:** Multi-class classification (3 classes).

**Target variable:** `behavioral_segment`
- **High Risk** — high social media usage, high "brain rot" index, low attention span
- **Balanced** — moderate behaviour across all dimensions
- **Focused** — controlled usage, high attention, strong academic habits

**Why is this a useful prediction task?**
Early identification of at-risk students enables educators and institutions to intervene before academic performance deteriorates. Knowing a student's behavioural profile from self-reported digital habits allows targeted support, e.g. screen-time counselling for High-Risk students or academic enrichment for Focused ones.

---

## 2. Dataset Description

| Property | Value |
|---|---|
| **Source** | [Kaggle — Student Social Media & Brain Rot Dataset](https://www.kaggle.com/datasets/nitikachandel95/student-social-media-impact-dataset) |
| **Generation** | Synthetically generated (Python / NumPy / Pandas), dependency-driven simulation |
| **Rows** | ~500,000 |
| **Columns** | 35+ (mix of numerical and categorical) |
| **File** | `data/global_student_digital_behavior_dataset.csv` |

### Feature Overview

| Category | Key Features |
|---|---|
| Demographics | `age`, `gender`, `country`, `development_level` |
| Socioeconomic | `family_income_usd`, `urban_rural`, `internet_access`, `internet_speed_mbps` |
| Education | `education_level`, `field_of_study`, `academic_motivation`, `online_learning_hours` |
| Social Media | `daily_social_media_hours`, `session_frequency_per_day`, `avg_session_length_minutes`, `late_night_usage_hours` |
| Content Mix | `educational_content_pct`, `entertainment_content_pct`, `short_video_pct`, `news_content_pct` |
| Engagement | `daily_likes`, `daily_comments`, `content_creation_hours` |
| Cognitive/Academic | `attention_span_minutes`, `study_hours_per_day`, `class_attendance_pct`, `productivity_score` |
| Psychological | `stress_level`, `anxiety_score`, `depression_score`, `avg_daily_sleep_hours` |
| Economic | `ad_exposure_per_day`, `impulse_purchases_per_month`, `monthly_digital_spending_usd` |

### Target Distribution (approximate, by design)

| Segment | Proportion |
|---|---|
| Balanced | ~40% |
| High Risk | ~35% |
| Focused | ~25% |

---

## 3. Preprocessing Approach

> **Split-first discipline strictly enforced.**
> All statistics (medians, IQR bounds, encoder mappings, scaler parameters) are derived **exclusively from `X_train`** and then applied identically to `X_val` and `X_test`.

### 3.1 Train / Val / Test Split
Stratified 80 / 10 / 10 split using two sequential `train_test_split` calls with `stratify=y` to preserve class ratios. `random_state=42` for reproducibility.

### 3.2 Missing Values
- **Numerical columns:** imputed with **median** (preferred over mean due to skewed distributions such as `family_income_usd` and `daily_social_media_hours`).
- **Categorical columns:** imputed with **mode** or `"Unknown"` if no mode is available.
- **Target column:** rows with missing target are dropped before splitting.
- All fill values computed on `X_train` only, then applied to `X_val` and `X_test`.

### 3.3 Outlier Treatment
**Method:** IQR Winsorising — values below `Q1 − 1.5·IQR` or above `Q3 + 1.5·IQR` are **capped** (not removed).

**Rationale:** With ~500k rows, removing outliers would discard potentially valid extreme-usage students (e.g. 14+ hours/day on social media). Capping preserves all rows while bounding the influence of extreme values on model training.

IQR bounds are computed on `X_train` only and applied to all three splits.

### 3.4 Encoding
| Column type | Strategy | Reason |
|---|---|---|
| Binary (2 unique values) | `LabelEncoder` | Minimal encoding overhead |
| Nominal categorical (≤15 unique) | One-Hot (`drop_first=True`) | Avoids multicollinearity |
| High-cardinality (>15 unique) | Dropped | Prevents feature explosion; target-encoding can be added in HW2 |

Column alignment after OHE: training-set column list is stored as `ohe_template`; val/test sets are aligned to this template (missing dummy columns filled with 0).

### 3.5 Feature Scaling
**StandardScaler** (zero mean, unit variance) fitted on `X_train` only, applied to all three splits.

Chosen over MinMaxScaler because:
- The neural network benefits from zero-centred inputs.
- XGBoost is tree-based and scale-invariant, but consistent scaling simplifies the shared pipeline.

The fitted scaler is saved as `models/scaler.pkl` for reuse in Homework 2.

---

## 4. Feature Engineering

Three new domain-informed features are created **before** imputation (so raw column values are still intact):

### Feature 1 — `digital_intensity`
```
digital_intensity = daily_social_media_hours × session_frequency_per_day
```
Captures the **multiplicative load** of both duration and frequency of social media use. A student who uses social media for 6 hours in 12 short sessions has a very different cognitive profile from one who uses it for 6 hours in 2 long sessions — yet raw hours alone cannot distinguish them.

### Feature 2 — `sleep_debt_score`
```
sleep_debt_score = 1 / (avg_daily_sleep_hours + 1)
```
A bounded, non-linear recency-style score. A student sleeping 4 hours scores ≈ 0.20; one sleeping 8 hours scores ≈ 0.11. The `+1` prevents division-by-zero. This captures the diminishing returns of each additional hour of sleep in a compact [0, 1] range that is directly interpretable by both models.

### Feature 3 — `productive_ratio`
```
productive_ratio = online_learning_hours / (daily_social_media_hours + 1)
```
Measures the **fraction of digital time spent constructively**. A student with 3 online learning hours and 3 social media hours scores 0.75; one with 0.5 learning hours and 10 social media hours scores 0.045. This directly operationalises the "offsetting hypothesis" raised in the dataset description.

All three features are **added alongside** original columns — originals are not replaced.

---

## 5. PCA Insights

PCA is run on the **scaled training set** after all preprocessing steps. Plots saved to `models/`.

### Scree Plot (`models/pca_scree.png`)
The scree plot reveals how many components are needed to capture 90% of total variance. Based on the dataset's design (many interdependent behavioural variables), the first few components typically capture a disproportionate share of variance, reflecting the underlying causal chain:

> `Digital Access → Social Media Behaviour → Cognitive/Psychological Outcomes`

### Loadings Heatmap (`models/pca_loadings.png`)
The top contributors to **PC1** are expected to be variables directly tied to social media intensity: `digital_intensity`, `daily_social_media_hours`, `short_video_pct`, and `late_night_usage_hours`. These reflect the "brain rot" axis — the primary dimension of variation in the dataset.

**PC2** is expected to be driven by academic/cognitive outputs: `study_hours_per_day`, `class_attendance_pct`, `attention_span_minutes`, and `productive_ratio` — a dimension orthogonal to raw usage intensity.

The engineered features (`digital_intensity`, `sleep_debt_score`, `productive_ratio`) appearing prominently in the loadings validates the feature engineering choices.

### 2-D Scatter (`models/pca_scatter.png`)
The projection onto PC1 and PC2 is expected to show rough separation between **High Risk** and **Focused** students along PC1, with **Balanced** students occupying the middle region. This confirms that the two principal axes align with meaningful real-world behavioural gradients rather than noise.

---

## 6. Model Comparison

> Full numerical results are generated at runtime and saved to `models/model_comparison.csv`.
> The table below shows the structure; actual values depend on your run.

| Metric | XGBoost | Neural Network |
|---|---|---|
| Accuracy | — | — |
| Precision (macro) | — | — |
| Recall (macro) | — | — |
| F1 (macro) | — | — |
| ROC-AUC (macro) | — | — |

### Expected outcome discussion

On **tabular datasets of this structure**, XGBoost typically matches or outperforms a shallow feedforward network because:

1. Tree-based models naturally capture the non-linear, interaction-heavy relationships present in behavioural data without requiring careful architecture search.
2. With 500k rows, the neural network has enough data to learn, but the XGBoost's inductive bias (splitting on feature thresholds) is inherently well-suited to the ordinal and categorical nature of many features.
3. The tuned XGBoost (Task 6 RandomizedSearchCV) has an additional advantage over the fixed-architecture neural network.

Whether the neural network's loss curves show convergence or plateau depends on the effective batch dynamics; early stopping (patience=15) prevents overfitting.

**Do the top XGBoost features align with PCA loadings?**
Yes — features that dominate PC1 (digital intensity, social media hours, short video %) are expected to also rank highly in XGBoost feature importances, since both methods surface the most discriminative axes of variation.

---

## 7. Best Model Designation

The model with the higher **macro ROC-AUC on the test set** is designated `best_model.pkl`.

- If **XGBoost** wins → `models/best_model.pkl` is a copy of `models/classical_model.pkl` (a `joblib`-serialised `XGBClassifier`).
- If **Neural Network** wins → `models/best_model.pkl` is a copy of `models/neural_network.pt` (a PyTorch checkpoint dict).

The winning model is printed at the end of `main.py` and recorded in `models/model_comparison.csv`.

**This file is what Homework 2 will load directly** — do not delete or overwrite it after a successful run.

---

## 8. Installation & Execution

### Requirements
- Python 3.10+
- pip

### Step-by-step

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd hw1

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and place it at:
#    data/global_student_digital_behavior_dataset.csv
#    (or rename the downloaded file accordingly)

# 5. Run the full pipeline  (includes hyperparameter tuning — may take ~10 min)
python main.py

# 5b. Faster run without tuning
python main.py --no-tuning

# 6. (Optional) Start the FastAPI prediction server
uvicorn src.api:app --reload
# Then open: http://127.0.0.1:8000/docs
```

### Example API call (after starting the server)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "daily_social_media_hours": 8,
    "session_frequency_per_day": 15,
    "avg_daily_sleep_hours": 5,
    "study_hours_per_day": 1,
    "online_learning_hours": 0.5,
    "attention_span_minutes": 10,
    "stress_level": 8
  }'
```

Expected response:
```json
{
  "prediction": 0,
  "label": "High Risk",
  "probabilities": {
    "Balanced": 0.0821,
    "Focused": 0.0312,
    "High Risk": 0.8867
  }
}
```

### Output files after a successful run

```
models/
├── classical_model.pkl          # Trained XGBoost
├── neural_network.pt            # Trained PyTorch network
├── best_model.pkl               # Copy of the winning model
├── scaler.pkl                   # Fitted StandardScaler
├── label_encoder.pkl            # Target LabelEncoder
├── preprocessing_artifacts.pkl  # fill values, IQR bounds, encoders
├── pca_scree.png
├── pca_loadings.png
├── pca_scatter.png
├── nn_loss_curve.png
├── cm_xgboost.png
├── cm_neural_network.png
├── feature_importance.png
└── model_comparison.csv / .png
```

---

## Reproducibility

`random_state=42` is set in every call that accepts a seed:
- `train_test_split` (both splits)
- `XGBClassifier`
- `RandomizedSearchCV`
- `StratifiedKFold`
- `PCA`
- `torch.manual_seed`
- `np.random.seed`

Results are fully reproducible given the same dataset file.
