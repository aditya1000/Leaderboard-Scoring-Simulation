import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# ------------------------------
# Set Page Configuration
# ------------------------------
st.set_page_config(
    page_title="🏆 Leaderboard Score Sensitivity Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Title and Description
# ------------------------------
st.title("🏆 Leaderboard Score Sensitivity Analysis")

st.markdown("""
Welcome to the **Leaderboard Score Sensitivity Analysis** tool for the **Pediatric Sepsis Data Challenge**!

Adjust the **Weights** and **Metrics** using the controls in the sidebar. The main area will display the current leaderboard score and an interactive sensitivity analysis plot to show how changes in metrics affect the score.

**Leaderboard Score Formula:**

\[
  \t{Score} = (w_T * T) + (w_P * P) + (w_A * A) + (w_B * B) + (w_F * F) + (w_I * I) + (w_C * C) + (w_R * R)
\]

- **T:** TPR@FPR ≤ 0.20
- **P:** PPV
- **A:** AUC
- **B:** Balanced Accuracy
- **F:** F1 Score
- **I:** Normalized Inference Time (Penalty)
- **C:** Normalized Compute (Penalty)
- **R:** Parsimony
""")

# ------------------------------
# Sidebar: Adjust Weights and Metrics
# ------------------------------
st.sidebar.header("⚙️ Settings")

# ------------------------------
# Adjust Weights (Locked Display)
# ------------------------------
st.sidebar.subheader("🔧 Adjust Weights (Locked)")

# Define fixed weights
fixed_weights = {
    'w_T': 0.30,  # Weight for TPR@FPR ≤ 0.20
    'w_P': 0.20,  # Weight for PPV
    'w_A': 0.20,  # Weight for AUC
    'w_B': 0.15,  # Weight for Balanced Accuracy
    'w_F': 0.10,  # Weight for F1 Score
    'w_I': -0.05, # Penalty for Inference Time
    'w_C': -0.05, # Penalty for Compute Utilized
    'w_R': 0.05   # Weight for Parsimony
}

# Display fixed weights as compact text
for key, value in fixed_weights.items():
    st.sidebar.markdown(f"**{key}:** {value}")

# ------------------------------
# Adjust Metrics (Adjustable Sliders)
# ------------------------------
st.sidebar.subheader("📊 Adjust Metrics")

# Define metric sliders
T_val = st.sidebar.slider(
    "T (TPR@FPR ≤ 0.20)",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.01,
    key='T_val'
)

P_val = st.sidebar.slider(
    "P (PPV)",
    min_value=0.0,
    max_value=1.0,
    value=0.65,
    step=0.01,
    key='P_val'
)

A_val = st.sidebar.slider(
    "A (AUC)",
    min_value=0.0,
    max_value=1.0,
    value=0.85,
    step=0.01,
    key='A_val'
)

B_val = st.sidebar.slider(
    "B (Balanced Accuracy)",
    min_value=0.0,
    max_value=1.0,
    value=0.78,
    step=0.01,
    key='B_val'
)

F_val = st.sidebar.slider(
    "F (F1 Score)",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.01,
    key='F_val'
)

I_val = st.sidebar.slider(
    "I (Normalized Inference Time)",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.01,
    key='I_val'
)

C_val = st.sidebar.slider(
    "C (Normalized Compute)",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.01,
    key='C_val'
)

R_val = st.sidebar.slider(
    "R (Parsimony)",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.01,
    key='R_val'
)

# ------------------------------
# Function to Compute Score
# ------------------------------
def compute_score(T, P, A, B, F, I, C, R,
                 fixed_weights):
    """
    Compute the leaderboard score based on metrics and fixed weights.
    """
    return (fixed_weights['w_T'] * T) + \
           (fixed_weights['w_P'] * P) + \
           (fixed_weights['w_A'] * A) + \
           (fixed_weights['w_B'] * B) + \
           (fixed_weights['w_F'] * F) + \
           (fixed_weights['w_I'] * I) + \
           (fixed_weights['w_C'] * C) + \
           (fixed_weights['w_R'] * R)

# Compute the current score
score = compute_score(
    T=T_val,
    P=P_val,
    A=A_val,
    B=B_val,
    F=F_val,
    I=I_val,
    C=C_val,
    R=R_val,
    fixed_weights=fixed_weights
)

# ------------------------------
# Main Page: Display Score
# ------------------------------
st.markdown("## 🏅 Current Leaderboard Score")

# Display the numeric score prominently
score_col1, score_col2 = st.columns([1, 3])

with score_col1:
    st.markdown(f"### **{score:.4f}**")

with score_col2:
    st.progress(score)

## Explanation section
st.markdown("""
### How to Interpret This
- Increase T to see how capturing more critical positives at low FPR affects score.
- Increase P to boost PPV, ensuring flagged cases are truly high-risk.
- A, B, and F influence overall discriminative ability, balanced performance, and precision-recall tradeoffs.
- I and C penalize slow or resource-heavy models; increasing these reduces the final score.
- R rewards simpler (more parsimonious) models. Increasing R raises the final score.
""")

# ------------------------------
# Sensitivity Analysis Section
# ------------------------------
st.markdown("## 🔍 Sensitivity Analysis")

st.markdown("""
Select a metric to vary and observe how it affects the leaderboard score. This helps in understanding which metrics have the most significant impact on the overall score.
""")

# Dropdown to select which metric to vary
metric_to_vary = st.selectbox("Select a metric to vary:", ["T", "P", "A", "B", "F", "I", "C", "R"])

# Generate data for the selected metric variation
var_range = np.linspace(0, 1, 100)
scores = []

for val in var_range:
    # Update the selected metric while keeping others constant
    current_metrics = {
        'T': T_val,
        'P': P_val,
        'A': A_val,
        'B': B_val,
        'F': F_val,
        'I': I_val,
        'C': C_val,
        'R': R_val
    }
    current_metrics[metric_to_vary] = val
    s = compute_score(
        T=current_metrics['T'],
        P=current_metrics['P'],
        A=current_metrics['A'],
        B=current_metrics['B'],
        F=current_metrics['F'],
        I=current_metrics['I'],
        C=current_metrics['C'],
        R=current_metrics['R'],
        fixed_weights=fixed_weights
    )
    scores.append(s)

# Create a DataFrame for plotting
df_line = pd.DataFrame({
    metric_to_vary: var_range,
    'Score': scores
})

# Create the line chart using Altair
line_chart = alt.Chart(df_line).mark_line(color='blue').encode(
    x=alt.X(f"{metric_to_vary}:Q", title=f"{metric_to_vary} Value"),
    y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0,1])),
    tooltip=[f"{metric_to_vary}:Q", 'Score:Q']
).properties(
    width=800,
    height=400
).interactive()

st.altair_chart(line_chart, use_container_width=True)

st.markdown("""
### ** Sensitivity Analysis Interpretation :**
- **Steep Slopes:** Indicate high sensitivity. Small changes in the selected metric lead to significant score variations.
- **Flat Slopes:** Indicate low sensitivity. Changes in the metric have minimal impact on the score.
- **Direction of Change:** 
  - **Positive Metrics (T, P, A, B, F, R):** Typically, increasing these improves the score.
  - **Penalty Metrics (I, C):** Increasing these reduces the score.
""")

