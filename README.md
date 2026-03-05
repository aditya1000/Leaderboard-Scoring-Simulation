# 🏆 Leaderboard Scoring — Pediatric Sepsis Data Challenge

This Streamlit application displays leaderboard results for the **Pediatric Sepsis Data Challenge** across two phases:

- **Final Phase Leaderboard** — the primary, current view based on the final evaluation pipeline.
- **Phase-1 Leaderboard Ranking** — the earlier phase ranking retained for reference.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run strmlit_LS3.py
```

Use the **Navigation** radio buttons in the sidebar to switch between views.

---

## 🏁 Final Phase Leaderboard

### Eligibility Criterion

Only submissions where **Sensitivity ≥ 0.80** qualify for ranked scoring. All other submissions are listed separately as **Low Sensitivity**, **Error**, or **Late**.

Submissions received **after July 18, 2025** are marked **Late** and excluded from the main ranking.

---

### 🧮 Scoring Pipeline

The Final Phase score is computed in three steps:

#### Step 1 — Weighted Score

A linear combination of six metrics using fixed factor loadings:

$$
\text{Weighted Score} = (0.6477 \times F_1) + (0.3447 \times \text{AUPRC}) + (0.8514 \times \text{Net Benefit}) - (0.8675 \times \text{ECE}) + (0.05 \times \text{Inference Speed}) + (0.05 \times \text{Parsimony Score})
$$

| Metric | Factor Loading | Direction |
|---|---|---|
| **F1** | 0.6477 | Higher is better |
| **AUPRC** | 0.3447 | Higher is better |
| **Net Benefit** | 0.8514 | Higher is better |
| **ECE** | −0.8675 | Lower is better (penalty) |
| **Inference Speed** | 0.05 | Higher is better |
| **Parsimony Score** | 0.05 | Higher is better (stored as 1 − raw parsimony) |

#### Step 2 — Min-Max Scaling

The weighted score components are standardised using pre-fitted center and scale parameters (`scale_params.json`):

$$
x_{\text{scaled}} = \frac{x - \text{center}}{\text{scale}}
$$

| Parameter | Values |
|---|---|
| Center | [0.1775, 0.2452, −0.1304, 0.1272, 0.0007] |
| Scale  | [0.0547, 0.0932, 0.4445, 0.1850, 0.0015] |

#### Step 3 — Z-Score Normalisation

The min-max scaled score is further normalised with a global z-score transform (`zscore_params.json`) to produce the final **Scaled Weighted Score** used for ranking:

$$
\text{Scaled Weighted Score} = \frac{x_{\text{scaled}} - \mu}{\sigma}
$$

| Parameter | Value |
|---|---|
| Center (μ) | ≈ 0 (−5.8683 × 10⁻¹⁶) |
| Scale (σ) | 1.9692 |

---

### 📊 Leaderboard Views

| View | Description |
|---|---|
| **✅ Complete Submissions** | All qualifying submissions (Sensitivity ≥ 0.8, on time), ranked by Scaled Weighted Score. Ties share the same dense rank. |
| **🏆 Best Performance per Team** | The single highest-scoring complete submission per team. |
| **⚠️ Flagged / Error / Late Submissions** | Submissions excluded from the main ranking (expandable panel). |
| **🗺 World Map** | Geographic view of best-per-team scores (complete submissions only). |

---

### 📋 Displayed Columns

Each complete submission row shows:

- **Rank** — Dense rank by Scaled Weighted Score (ties share the same rank)
- **Team name** — Team name with country flag
- **Affiliation**
- **Weighted Score** — Raw weighted linear combination (Step 1)
- **Scaled Weighted Score** — Final normalised score used for ranking (Step 3)
- **AUPRC** — Area Under the Precision-Recall Curve
- **Net Benefit** — Clinical utility metric
- **ECE** — Expected Calibration Error
- **Parsimony Score** — 1 − raw parsimony (feature efficiency; higher is better)
- **Inference Time**
- **F1, TP, FP, FN, TN, AUC** — Additional performance statistics
- **Status** — `Complete`, `Late`, `Low Sensitivity`, or `Error`
- **Submission time**

---

## 📘 Phase-1 Leaderboard Ranking (Reference)

The Phase-1 score uses a simpler weighted formula with adjustable sidebar sliders:

$$
\text{Score} = (w_A \times A) + (w_{Ap} \times A_p) + (w_{Nb} \times Nb) - (w_{ECE} \times ECE) - (w_I \times I)
$$

| Component | Description | Default Weight |
|---|---|---|
| **A** | AUC | 0.626 |
| **Ap** | AUPRC | 0.417 |
| **Nb** | Net Benefit | 0.974 |
| **ECE** | Expected Calibration Error (penalty) | 0.907 |
| **I** | Normalised Inference Time (penalty) | 0.05 |

Inference Time is normalised by the maximum inference time across all submissions before scoring.

---

## 🛠 Technical Requirements

- **Python 3.8+**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Altair**
- **PyDeck**

Install all dependencies:

```bash
pip install -r requirements.txt
```
