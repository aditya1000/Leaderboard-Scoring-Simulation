import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import json

# ------------------------------
# Set Page Configuration
# ------------------------------
st.set_page_config(
    page_title="🏆 Leaderboard Scoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Sidebar: Navigation
# ------------------------------
page = st.sidebar.radio("Navigation", ["Leaderboard Ranking", "Score Sensitivity Analysis"])

if page == "Leaderboard Ranking":
# =============================================================================
# Page 1: Leaderboard Ranking
# =============================================================================
    st.title("🏆 Leaderboard Ranking")
    st.markdown("""
    The leaderboard rankings are automatically loaded from the JSON file **leaderboard.json** in the root directory.
    
    **Expected JSON Format:**  
    ```json
    [
      {"team": "Team A", "score": 0.92},
      {"team": "Team B", "score": 0.85},
      {"team": "Team C", "score": 0.78}
    ]
    ```
    """)

    # Define the path to the JSON file in the root directory.
    json_file = "leaderboard.json"

    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # Create a DataFrame and validate the required columns.
        df_leaderboard = pd.DataFrame(data)
        if 'team' not in df_leaderboard.columns or 'score' not in df_leaderboard.columns:
            st.error("The JSON file must contain keys 'team' and 'score'.")
        else:
            df_leaderboard = df_leaderboard.sort_values("score", ascending=False)
            st.subheader("Leaderboard Ranking")
            st.table(df_leaderboard)

            # Create a bar chart of the leaderboard
            chart = alt.Chart(df_leaderboard).mark_bar().encode(
                x=alt.X("score:Q", title="Score"),
                y=alt.Y("team:N", sort="-x", title="Team")
            ).properties(width=600, height=400)
            st.altair_chart(chart, use_container_width=True)
    except FileNotFoundError:
        st.error(f"The file '{json_file}' was not found in the root directory. Please add the file and try again.")
    except Exception as e:
        st.error(f"An error occurred while reading the JSON file: {e}")

elif page == "Score Sensitivity Analysis":
# =============================================================================
# Page 2: Score Sensitivity Analysis
# =============================================================================
    # ------------------------------
    # Title and Description
    # ------------------------------
    st.title("🏆 Leaderboard Score Sensitivity Analysis")

    st.markdown("""
    Welcome to the **Leaderboard Score Sensitivity Analysis** tool for the **Pediatric Sepsis Data Challenge**!

    Adjust the **Weights** and **Metrics** using the controls in the sidebar. The main area will display the current leaderboard score and an interactive sensitivity analysis plot to show how changes in metrics affect the score.

    **Leaderboard Score Formula:**

\[
  \t{Score} = (w_A * A) + (w_Ap * Ap) + (w_Nb * Nb) - (w_E * ECE) + (w_I * I) + (w_C * C)
\]


- **A:** AUC
- **Ap:** AUPRC
- **Nb:** Net Benefit
- **ECE:** Expected Calibration Error            
- **I:** Normalized Inference Time (Penalty)
- **C:** Normalized Compute (Penalty)
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
    'w_A': 0.25,  # Weight for AUC
    'w_Ap': 0.35,  # Weight for AUPRC
    'w_Nb': 0.30,  # Weight for NB Score
    'w_ECE': -0.10,  # Weight for ECE
    'w_I': -0.05, # Penalty for Inference Time
    'w_C': -0.05, # Penalty for Compute Utilized
}

# Display fixed weights as compact text
for key, value in fixed_weights.items():
    st.sidebar.markdown(f"**{key}:** {value}")

    # ------------------------------
    # Adjust Metrics (Adjustable Sliders)
    # ------------------------------
    st.sidebar.subheader("📊 Adjust Metrics")

    # Define metric sliders

    A_val = st.sidebar.slider(
        "A (AUC)",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        key='A_val'
    )

    Ap_val = st.sidebar.slider(
        "AUPRC (area under precision recall curve)",
        min_value=0.0,
        max_value=1.0,
        value=0.78,
        step=0.01,
        key='Ap_val'
    )

    Nb_val = st.sidebar.slider(
        "Nb (net benefit)",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01,
        key='Nb_val'
    )

    ECE_val = st.sidebar.slider(
        "ECE (Expected Calibration Error)",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01,
        key='ECE_val'
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


# ------------------------------
# Function to Compute Score
# ------------------------------
def compute_score(A, Ap, Nb, ECE,I, C, 
                 fixed_weights):
    """
    Compute the leaderboard score based on metrics and fixed weights.
    """
    return (fixed_weights['w_A'] * A) + \
           (fixed_weights['w_Ap'] * Ap) + \
           (fixed_weights['w_Nb'] * Nb) + \
           (fixed_weights['w_ECE'] * ECE) + \
           (fixed_weights['w_I'] * I) + \
           (fixed_weights['w_C'] * C)

# Compute the current score
score = compute_score(
   
    A=A_val,
    Ap=Ap_val,
    Nb=Nb_val,
    ECE= ECE_val,
    I=I_val,
    C=C_val,
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
    - **A (AUC):** Measures the discrimination ability of the model across all thresholds (how well the model distinguishes between positive and negative classes).
    - **AUPRC :** Focuses on precision and recall for the positive class (mortality), especially valuable in imbalanced datasets.
    - **Nb (Net Benefit):** Measures the clinical utility, evaluate if benefits of using a model outweigh its potential harms.
    - **ECE (Expected Calibration Error):** Measures the calibration of predicted probabilities, ensuring that the predicted probability matches the true outcome.
    - **I (Normalized Inference Time):** Penalty for high inference time, ensuring that the model’s prediction time remains efficient in real-world deployment.
    - **C (Normalized Compute):** Penalty for high compute cost, considering the resources needed for running the model (important in resource-constrained environments).
    """)

# ------------------------------
# Sensitivity Analysis Section
# ------------------------------
st.markdown("## 🔍 Sensitivity Analysis")

st.markdown("""
Select a metric to vary and observe how it affects the leaderboard score. This helps in understanding which metrics have the most significant impact on the overall score.
""")

# Dropdown to select which metric to vary
metric_to_vary = st.selectbox("Select a metric to vary:", [ "A", "Ap", "Nb", "ECE", "I", "C"])

# Generate data for the selected metric variation
var_range = np.linspace(0, 1, 100)
scores = []

for val in var_range:
    # Update the selected metric while keeping others constant
    current_metrics = {
        'A': A_val,
        'Ap': Ap_val,
        'Nb': Nb_val,
        'ECE': ECE_val,
        'I': I_val,
        'C': C_val
    }
    current_metrics[metric_to_vary] = val
    s = compute_score(
        A=current_metrics['A'],
        Ap=current_metrics['Ap'],
        Nb=current_metrics['Nb'],
        ECE=current_metrics['ECE'],
        I=current_metrics['I'],
        C=current_metrics['C'],
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
    ### Sensitivity Analysis Interpretation :
    - **Steep Slopes:** Indicate high sensitivity. Small changes in the selected metric lead to significant score variations.
    - **Flat Slopes:** Indicate low sensitivity. Changes in the metric have minimal impact on the score.
    - **Direction of Change:** 
    - **Positive Metrics (A, Ap, Nb):** Typically, increasing these improves the score.
    - **Penalty Metrics (ECE, I, C):** Increasing these reduces the score.
    """)

