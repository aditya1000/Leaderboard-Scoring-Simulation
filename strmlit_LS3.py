import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import json
import os

# ------------------------------
# Set Page Configuration
# ------------------------------
st.set_page_config(
    page_title="🏆 Leaderboard Scoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Sidebar: Common Score Weights
# ------------------------------
st.sidebar.subheader("🔧 Score Weights")

# Default (fixed) weights for each metric
fixed_weights = {
    'w_A': 0.25,   # Weight for AUC
    'w_Ap': 0.35,  # Weight for AUPRC
    'w_Nb': 0.30,  # Weight for Net Benefit
    'w_ECE': 0.10, # Weight for ECE (Penalty)
    'w_I': 0.05,   # Weight for Inference Time (Penalty)
    'w_C': 0.05,   # Weight for Compute (Penalty)
}

w_A = st.sidebar.slider("w_A (Weight for AUC)", 0.0, 1.0, fixed_weights['w_A'], 0.01, key="w_A")
w_Ap = st.sidebar.slider("w_Ap (Weight for AUPRC)", 0.0, 1.0, fixed_weights['w_Ap'], 0.01, key="w_Ap")
w_Nb = st.sidebar.slider("w_Nb (Weight for Net Benefit)", 0.0, 1.0, fixed_weights['w_Nb'], 0.01, key="w_Nb")
w_ECE = st.sidebar.slider("w_ECE (Weight for ECE - Penalty)", 0.0, 0.2, fixed_weights['w_ECE'], 0.01, key="w_ECE")
w_I = st.sidebar.slider("w_I (Weight for Inference Time - Penalty)", 0.0, 0.2, fixed_weights['w_I'], 0.01, key="w_I")
w_C = st.sidebar.slider("w_C (Weight for Compute - Penalty)", 0.0, 0.2, fixed_weights['w_C'], 0.01, key="w_C")

dynamic_weights = {
    'w_A': w_A,
    'w_Ap': w_Ap,
    'w_Nb': w_Nb,
    'w_ECE': w_ECE,
    'w_I': w_I,
    'w_C': w_C,
}



# ------------------------------
# Sidebar: Navigation
# ------------------------------
#page = st.sidebar.radio("Navigation", ["Score Sensitivity Analysis", "Leaderboard Ranking"])
page = st.sidebar.radio("Navigation", ["Score Sensitivity Analysis"])


if page == "Leaderboard Ranking":
    # =============================================================================
    # Page 1: Leaderboard Ranking
    # =============================================================================
    st.title("🏆 Leaderboard Ranking")
    json_file = "leaderboard_with_info.json"

    def compute_score_json(team_data, weights):
        # Compute overall score using the given weights.
        # Here, team_data["Compute"] is assumed to be a list and index 1 is used.
        return (weights['w_A'] * team_data["AUC"]) + \
               (weights['w_Ap'] * team_data["AUPRC"]) + \
               (weights['w_Nb'] * team_data["Net Benefit"]) - \
               (weights['w_ECE'] * team_data["ECE"]) - \
               (weights['w_I'] * team_data["Inference Time"]) - \
               (weights['w_C'] * team_data["Compute"][1])
    
    # Define a simple mapping of country names to flag emojis.
    flag_dict = {
    "Cameroon": "🇨🇲",
    "Canada": "🇨🇦",
    "China": "🇨🇳",
    "Germany": "🇩🇪",
    "India": "🇮🇳",
    "Kenya": "🇰🇪",
    "Pakistan": "🇵🇰",
    "Romania": "🇷🇴",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Switzerland": "🇨🇭",
    "Uganda": "🇺🇬",
    "UK": "🇬🇧",
    "USA": "🇺🇸"
    }

    def country_to_flag(country_str):
        # Split on commas and trim spaces; convert each country to its flag.
        countries = [c.strip() for c in country_str.split(",")]
        flags = [flag_dict.get(c, c) for c in countries]
        return " ".join(flags)
    
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # Convert JSON data to DataFrame.
        df_leaderboard = pd.DataFrame(data)
        
        # Check that required keys exist.
        required_keys = {"team", "AUC", "AUPRC", "Net Benefit", "ECE", "Inference Time", "Compute", "Institution", "Country"}
        if not required_keys.issubset(df_leaderboard.columns):
            st.error("The JSON file must contain the keys: " + ", ".join(required_keys))
        else:
            # Compute score for each team using dynamic weights.
            df_leaderboard["Score"] = df_leaderboard.apply(lambda row: compute_score_json(row, dynamic_weights), axis=1)
            
            # Sort by score (descending).
            df_leaderboard = df_leaderboard.sort_values("Score", ascending=False)
            df_leaderboard["Rank"] = range(1, len(df_leaderboard) + 1)
            
            # Split the Compute column into two separate columns.
            df_leaderboard["Compute_1"] = df_leaderboard["Compute"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
            df_leaderboard["Compute_2"] = df_leaderboard["Compute"].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)
            
            # Create a new column "Country_Flag" by converting the Country field into flag(s).
            df_leaderboard["Country_Flag"] = df_leaderboard["Country"].apply(country_to_flag)
            
            # Create a combined field "teamInfo" with newlines.
            df_leaderboard["teamInfo"] = df_leaderboard["team"].str.strip() + "\n(" + \
                                          df_leaderboard["Institution"].str.strip() + ",\n" + \
                                          df_leaderboard["Country_Flag"].str.strip() + ")"
            
            # Remove duplicate entries based on the team field.
            df_leaderboard = df_leaderboard.drop_duplicates(subset=["team"], keep="first")
            
            # Define the display columns (omitting the "Rank" column).
            #display_cols = ["team","Country_Flag", "Score", "AUC", "AUPRC", "Net Benefit", "ECE", "Inference Time", "Compute_1", "Compute_2"]
            
            st.subheader("Full Leaderboard Ranking")
            
            # Define the internal display columns.
            display_cols = ["Rank", "team", "Country_Flag", "Score", "AUC", "AUPRC", "Net Benefit", "ECE", "Inference Time", "Compute_1", "Compute_2"]

            # Mapping from your internal column names to the names you want to show.
            rename_dict = {
                "team": "Team Name",
                "Country_Flag": "Country",
                "Score": "Score",
                "AUC": "AUC",
                "AUPRC": "AUPRC",
                "Net Benefit": "Net Benefit",
                "ECE": "ECE",
                "Inference Time": "Inference Time",
                "Compute_1": "Memory Usage (MB)",
                "Compute_2": "CPU time (s)"
            }

            # Create a new DataFrame for display:
            df_display = df_leaderboard[display_cols].rename(columns=rename_dict).reset_index(drop=True)

            # Replace the index with blank strings so it doesn't show any row numbers.
            #df_display.index = [''] * len(df_display)

            # Display the DataFrame with st.dataframe().
            st.dataframe(df_display, hide_index=True)
            
            # Reset index and style the DataFrame to hide the index and preserve newlines.
            #styled_df = df_leaderboard[display_cols].reset_index(drop=True).style.set_properties(
            #    subset=["team"], **{'white-space': 'pre-wrap'}
            #).hide(axis="index")
            #st.write(styled_df.to_html(), unsafe_allow_html=True)
            
            # Create a bar chart showing team scores using teamInfo as the label.
            chart = alt.Chart(df_leaderboard).mark_bar().encode(
                x=alt.X("Score:Q", title="Score"),
                y=alt.Y("team:N", sort="-x", title="Team"),
                tooltip=display_cols
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
    st.title("🏆 Leaderboard Score Sensitivity Analysis")
    
    st.markdown("""
    Welcome to the **Leaderboard Score Sensitivity Analysis** tool for the **Pediatric Sepsis Data Challenge**!

    Adjust the **Weights** using the controls in the sidebar and the **Metrics** below. The main area will display the current leaderboard score and an interactive sensitivity analysis plot to show how changes in metrics affect the score.

    **Leaderboard Score Formula:**

    \[
      Score = (w_A * A) + (w_Ap * Ap) + (w_Nb * Nb) - (w_ECE * ECE) - (w_I * I) - (w_C * C)
    \]

    - **A:** AUC  
    - **Ap:** AUPRC  
    - **Nb:** Net Benefit  
    - **ECE:** Expected Calibration Error (Penalty)            
    - **I:** Normalized Inference Time (Penalty)  
    - **C:** Normalized Compute (Penalty)
    """)

    st.sidebar.subheader("📊 Adjust Metrics")
    A_val = st.sidebar.slider("A (AUC)", 0.0, 1.0, 0.85, 0.01, key="ms_A_val")
    Ap_val = st.sidebar.slider("AUPRC", 0.0, 1.0, 0.78, 0.01, key="ms_Ap_val")
    Nb_val = st.sidebar.slider("Net Benefit", 0.0, 1.0, 0.60, 0.01, key="ms_Nb_val")
    ECE_val = st.sidebar.slider("ECE", 0.0, 1.0, 0.60, 0.01, key="ms_ECE_val")
    I_val = st.sidebar.slider("Inference Time", 0.0, 1.0, 0.25, 0.01, key="ms_I_val")
    C_val = st.sidebar.slider("Compute", 0.0, 1.0, 0.25, 0.01, key="ms_C_val")

    def compute_score(A, Ap, Nb, ECE, I, C, weights):
        return (weights['w_A'] * A) + \
               (weights['w_Ap'] * Ap) + \
               (weights['w_Nb'] * Nb) - \
               (weights['w_ECE'] * ECE) - \
               (weights['w_I'] * I) - \
               (weights['w_C'] * C)

    score_dynamic = compute_score(A=A_val, Ap=Ap_val, Nb=Nb_val, ECE=ECE_val, I=I_val, C=C_val, weights=dynamic_weights)

    st.markdown("## 🏅 Current Leaderboard Score (Dynamic Weights)")
    normalized_score = min(max(score_dynamic, 0.0), 1.0)
    score_col1, score_col2 = st.columns([1, 3])
    with score_col1:
        st.markdown(f"### **{score_dynamic:.4f}**")
    with score_col2:
        st.progress(normalized_score)

    # Display the dynamic weights as a bar chart
    st.markdown("### Dynamic Weights Breakdown")
    weights_df = pd.DataFrame({
        "Metric": list(dynamic_weights.keys()),
        "Weight": list(dynamic_weights.values())
    })
    st.bar_chart(weights_df.set_index("Metric"))


    # ------------------------------
    # Function to Compute Score using Fixed Weights
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

    # ------------------------------
    # Function to Compute Score using dynamic weights
    # ------------------------------
    def compute_score_dynamic(A, Ap, Nb, ECE, I, C, weights):
        """
        Compute the leaderboard score dynamically based on adjustable weights.
        """
        return (weights['w_A'] * A) + \
            (weights['w_Ap'] * Ap) + \
            (weights['w_Nb'] * Nb) + \
            (weights['w_ECE'] * ECE) + \
            (weights['w_I'] * I) + \
            (weights['w_C'] * C)


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


    # Compute the current score with dynamic weights
    score_dynamic = compute_score_dynamic(
        A=A_val,
        Ap=Ap_val,
        Nb=Nb_val,
        ECE=ECE_val,
        I=I_val,
        C=C_val,
        weights=dynamic_weights
    )
    # ------------------------------
    # Main Page: Display Score with Dynamic Weights
    # ------------------------------
    st.markdown("## 🏅 Current Leaderboard Score (Dynamic Weights)")

    # Normalize the score to fall within [0.0, 1.0] for the progress bar
    normalized_score = min(max(score_dynamic, 0.0), 1.0)

    # Display the numeric score and progress bar
    score_col1, score_col2 = st.columns([1, 3])

    with score_col1:
        st.markdown(f"### **{score_dynamic:.4f}**")

    with score_col2:
        st.progress(normalized_score)


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
    # Sensitivity Analysis with Dynamic Weights
    # ------------------------------
    st.markdown("## 🔍 Sensitivity Analysis with Dynamic Weights")

    # Define ranges dynamically based on the sidebar sliders
    metric_ranges = {
        "A": (0.0, 1.0),  # AUC range
        "Ap": (0.0, 1.0),  # AUPRC range
        "Nb": (0.0, 1.0),  # Net Benefit range
        "ECE": (0.0, 1.0),  # ECE range
        "I": (0.0, 1.0),  # Inference Time range
        "C": (0.0, 1.0),  # Compute range
    }

    # Dropdown to select which metric to vary
    metric_to_vary = st.selectbox("Select a metric to vary:", ["A", "Ap", "Nb", "ECE", "I", "C"])

    # Generate range dynamically for the selected metric
    var_range = np.linspace(metric_ranges[metric_to_vary][0], metric_ranges[metric_to_vary][1], 100)

    # Generate scores for sensitivity analysis
    scores_dynamic = []

    # ------------------------------
    # Function to Compute Score using dynamic weights
    # ------------------------------
    def compute_score_dynamic(A, Ap, Nb, ECE, I, C, weights):
        """
        Compute the leaderboard score dynamically based on adjustable weights.
        """
        return (weights['w_A'] * A) + \
            (weights['w_Ap'] * Ap) + \
            (weights['w_Nb'] * Nb) + \
            (weights['w_ECE'] * ECE) + \
            (weights['w_I'] * I) + \
            (weights['w_C'] * C)


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
        s_dynamic = compute_score_dynamic(
            A=current_metrics['A'],
            Ap=current_metrics['Ap'],
            Nb=current_metrics['Nb'],
            ECE=current_metrics['ECE'],
            I=current_metrics['I'],
            C=current_metrics['C'],
            weights=dynamic_weights
        )
        scores_dynamic.append(s_dynamic)

    # Create a DataFrame for plotting
    df_line_dynamic = pd.DataFrame({
        metric_to_vary: var_range,
        'Score': scores_dynamic
    })

    # Create the line chart using Altair
    line_chart_dynamic = alt.Chart(df_line_dynamic).mark_line(color='green').encode(
        x=alt.X(f"{metric_to_vary}:Q", title=f"{metric_to_vary} Value"),
        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[min(scores_dynamic), max(scores_dynamic)])),
        tooltip=[f"{metric_to_vary}:Q", 'Score:Q']
    ).properties(
        width=800,
        height=400
    ).interactive()

    st.altair_chart(line_chart_dynamic, use_container_width=True)

    st.markdown("""
    ### Sensitivity Analysis Interpretation :
    - **Steep Slopes:** Indicate high sensitivity. Small changes in the selected metric lead to significant score variations.
    - **Flat Slopes:** Indicate low sensitivity. Changes in the metric have minimal impact on the score.
    - **Direction of Change:** 
    - **Positive Metrics (A, Ap, Nb):** Typically, increasing these improves the score.
    - **Penalty Metrics (ECE, I, C):** Increasing these reduces the score.
    """)

