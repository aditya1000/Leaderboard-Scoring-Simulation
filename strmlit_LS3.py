import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import json
import os
import pydeck as pdk

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
option = st.sidebar.radio("Navigation", ["Final Phase Leaderboard" ,"Phase-1 Leaderboard Ranking"]) #,"Score Sensitivity Analysis"])
#page = st.sidebar.radio("Navigation", ["Score Sensitivity Analysis"])



if option == "Phase-1 Leaderboard Ranking":
    # =============================================================================
    # Page 1: Leaderboard Ranking
    # =============================================================================
    st.title("🏆 Phase-1 Leaderboard Ranking")

    # ------------------------------
    # Sidebar: Common Score Weights
    # ------------------------------
    st.sidebar.subheader("🔧 Score Weights")

    # Default (fixed) weights for each metric
    fixed_weights = {
        'w_A': 0.626,   # Weight for AUC
        'w_Ap': 0.417,  # Weight for AUPRC
        'w_Nb': 0.974,  # Weight for Net Benefit
        'w_ECE': 0.907, # Weight for ECE (Penalty)
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
        'w_I': w_I ,
        'w_C': w_C,
    }


    with st.expander("Formula Details"):
        st.markdown("""
    
        **Leaderboard Score Formula:**
        \[
        Score = (w_A * A) + (w_Ap * Ap) + (w_Nb * Nb) - (w_ECE * ECE) - (w_I * I)
        \]
        - **A:** AUC  
        - **Ap:** AUPRC  
        - **Nb:** Net Benefit  
        - **ECE:** Expected Calibration Error (Penalty)            
        - **I:** Normalized Inference Time (Penalty)  
        """)
    
    json_file = "leaderboard_with_info.json"

    def compute_score_json(team_data, weights):
        # Compute overall score using the given weights.
        # Here, team_data["Compute"] is assumed to be a list and index 1 is used.
        return (weights['w_A'] * team_data["AUC"]) + \
               (weights['w_Ap'] * team_data["AUPRC"]) + \
               (weights['w_Nb'] * team_data["Net Benefit"]) - \
               (weights['w_ECE'] * team_data["ECE"]) - \
               (weights['w_I'] * team_data["Norm_inf"]) #- \
               #(weights['w_C'] * team_data["Compute"][1])

     # Predefined mapping from country name to approximate lat/lon.
    country_latlon = {
        "Cameroon": {"lat": 7.3697, "lon": 12.3547},
        "Canada": {"lat": 56.1304, "lon": -106.3468},
        "China": {"lat": 35.8617, "lon": 104.1954},
        "Germany": {"lat": 51.1657, "lon": 10.4515},
        "India": {"lat": 20.5937, "lon": 78.9629},
        "Kenya": {"lat": -0.0236, "lon": 37.9062},
        "Pakistan": {"lat": 30.3753, "lon": 69.3451},
        "Romania": {"lat": 45.9432, "lon": 24.9668},
        "South Africa": {"lat": -30.5595, "lon": 22.9375},
        "South Korea": {"lat": 35.9078, "lon": 127.7669},
        "Switzerland": {"lat": 46.8182, "lon": 8.2275},
        "Uganda": {"lat": 1.3733, "lon": 32.2903},
        "UK": {"lat": 55.3781, "lon": -3.4360},
        "USA": {"lat": 37.0902, "lon": -95.7129},
    }
    
    def get_lat_lon(country_str):
        # Assume country_str may contain multiple countries separated by commas; take the first.
        country = country_str.split(",")[0].strip()
        return country_latlon.get(country, {"lat": None, "lon": None})

    
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
            
            # Compute the maximum inference time in the DataFrame
            max_inf_time = df_leaderboard["Inference Time"].max()

            # Compute the normalized inference time, ensuring no division by zero.
            df_leaderboard["Norm_inf"] = df_leaderboard["Inference Time"].apply(lambda x: x / max_inf_time if max_inf_time else 0)
            
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
            
            # Create a new column "primaryCountry" that takes only the first country from the "Country" field.
            df_leaderboard["primaryCountry"] = df_leaderboard["Country"].apply(lambda x: x.split(",")[0].strip())


            
            # --- Create lat and lon columns using the first country in the "Country" field.
            df_leaderboard["lat"] = df_leaderboard["Country"].apply(lambda x: get_lat_lon(x)["lat"])
            df_leaderboard["lon"] = df_leaderboard["Country"].apply(lambda x: get_lat_lon(x)["lon"])
            
            # Remove duplicate entries based on the team field.
            df_leaderboard = df_leaderboard.drop_duplicates(subset=["team"], keep="first")
            
            # Define the display columns (omitting the "Rank" column).
            #display_cols = ["team","Country_Flag", "Score", "AUC", "AUPRC", "Net Benefit", "ECE", "Inference Time", "Compute_1", "Compute_2"]
            
            st.subheader("Full Leaderboard Ranking")
            
            # Define the internal display columns.
            display_cols = ["Rank", "team", "Country_Flag", "Score", "AUC", "AUPRC", "Net Benefit", "ECE", "Norm_inf"] #, "Compute_1", "Compute_2"]
            #display_cols = ["Rank", "team", "Score", "AUC", "AUPRC", "Net Benefit", "ECE", "Norm_inf"] #, "Compute_1", "Compute_2"]

            # Mapping from your internal column names to the names you want to show.
            rename_dict = {
                "team": "Team Name",
                "Country_Flag": "Location",
                "Score": "Score",
                "AUC": "AUC",
                "AUPRC": "AUPRC",
                "Net Benefit": "Net Benefit",
                "ECE": "ECE",
                "Norm_inf": "Normalized Inference Time"#,
                #"Compute_1": "Memory Usage (MB)",
                #"Compute_2": "CPU time (s)"
            }

            # Create a new DataFrame for display:
            df_display = df_leaderboard[display_cols].rename(columns=rename_dict).reset_index(drop=True)

            # Replace the index with blank strings so it doesn't show any row numbers.
            #df_display.index = [''] * len(df_display)

            # Display the DataFrame with st.dataframe().
            st.dataframe(df_display, hide_index=True, use_container_width=True)
            
            # Reset index and style the DataFrame to hide the index and preserve newlines.
            #styled_df = df_leaderboard[display_cols].reset_index(drop=True).style.set_properties(
            #    subset=["team"], **{'white-space': 'pre-wrap'}
            #).hide(axis="index")
            #st.write(styled_df.to_html(), unsafe_allow_html=True)
            
            
            
            # Instead of the bar chart, display a world map with team locations.
            # Create a PyDeck Scatterplot layer.
            # Now, group the DataFrame by "Country" so that one marker represents all teams in that country.
            # The tooltip will list all teamInfo values and their scores.
            # Group by country so that one marker represents all teams in that country.
            # Round the Score values to 2 decimals.
            df_leaderboard["Score"] = df_leaderboard["Score"].apply(lambda s: round(s, 2) if pd.notnull(s) else s)

            # Group the DataFrame by Country so that one marker represents all teams in that country.
            # For each country, we aggregate the team info and scores into one string.
            # Group by the "primaryCountry" so that one marker represents all teams from that country.
            grouped = df_leaderboard.groupby("primaryCountry").apply(
                lambda x: pd.Series({
                    "teams_scores": "<br/>".join(
                        [f"{row['teamInfo']}: {round(row['Score'], 2)}" for _, row in x.iterrows()]
                    ),
                    "lat": x["lat"].iloc[0],
                    "lon": x["lon"].iloc[0],
                    "primaryCountry": x["primaryCountry"].iloc[0]
                })
            ).reset_index(drop=True)

            # Create a PyDeck Scatterplot layer.
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=grouped,
                get_position=["lon", "lat"],
                get_fill_color="[0, 128, 200, 160]",
                get_radius=200000,  # Adjust marker radius as needed.
                pickable=True,
            )

            # Define an initial view that shows the world.
            view_state = pdk.ViewState(
                latitude=20,
                longitude=0,
                zoom=1,
                pitch=0
            )

            # Define an HTML tooltip that shows the country and, side by side, the teams and their scores.
            tooltip = {
                "html": """
                <div style="display: flex; flex-direction: row; justify-content: space-between; font-size: 12px;">
                <div style="margin-right: 10px;"> {primaryCountry}</div>
                <div style="margin-left: 10px;"><b>Teams & Scores:</b><br/>{teams_scores}</div>
                </div>
                """
            }

            # Create the Deck instance.
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip
            )

            st.subheader("Teams on the world map")
            st.pydeck_chart(deck)

            #Create a bar chart showing team scores using teamInfo as the label.
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
        
elif option == "Final Phase Leaderboard":
    st.title("🏁 Final Phase Leaderboard")

    import pandas as pd
    import json
    import pydeck as pdk
    import altair as alt

    # Predefined lat/lon and flag dictionaries
    country_latlon = {
        "Cameroon": {"lat": 7.3697, "lon": 12.3547},
        "Canada": {"lat": 56.1304, "lon": -106.3468},
        "China": {"lat": 35.8617, "lon": 104.1954},
        "Germany": {"lat": 51.1657, "lon": 10.4515},
        "India": {"lat": 20.5937, "lon": 78.9629},
        "Kenya": {"lat": -0.0236, "lon": 37.9062},
        "Pakistan": {"lat": 30.3753, "lon": 69.3451},
        "Romania": {"lat": 45.9432, "lon": 24.9668},
        "South Africa": {"lat": -30.5595, "lon": 22.9375},
        "South Korea": {"lat": 35.9078, "lon": 127.7669},
        "Switzerland": {"lat": 46.8182, "lon": 8.2275},
        "Uganda": {"lat": 1.3733, "lon": 32.2903},
        "United Kingdom": {"lat": 55.3781, "lon": -3.4360},
        "USA": {"lat": 37.0902, "lon": -95.7129},
    }
    flag_dict = {
        "Cameroon": "🇨🇲", "Canada": "🇨🇦", "China": "🇨🇳", "Germany": "🇩🇪",
        "India": "🇮🇳", "Kenya": "🇰🇪", "Pakistan": "🇵🇰", "Romania": "🇷🇴",
        "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Switzerland": "🇨🇭",
        "Uganda": "🇺🇬", "United Kingdom": "🇬🇧", "USA": "🇺🇸"
    }

    # Manual geo lookup
    def get_lat_lon_manual(country_field):
        if pd.isna(country_field):
            return {"lat": None, "lon": None}
        country = str(country_field).split(",")[0].strip()
        return country_latlon.get(country, {"lat": None, "lon": None})

    # Load parameter files
    with open("factor_loadings.json", "r") as f:
        factor_loadings = json.load(f)
    with open("scale_params.json", "r") as f:
        scale_params = json.load(f)
    with open("zscore_params.json", "r") as f:
        zscore_params = json.load(f)

    # Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("ℹ️ **Leaderboard weighted scoring only includes models with Sensitivity ≥ 0.8. Submissions after the 19th are marked Late.**")
    st.sidebar.markdown("### ⚙️ Scaling Parameters")
    st.sidebar.markdown("**Fixed Factor Loadings:**")
    st.sidebar.json(factor_loadings)
    st.sidebar.markdown("**Min-Max Scaling Parameters:**")
    st.sidebar.markdown(f"**Center:** `{scale_params['center']}`")
    st.sidebar.markdown(f"**Scale:** `{scale_params['scale']}`")
    st.sidebar.markdown("**Z-Score Scaling Parameters:**")
    st.sidebar.markdown(f"**Center:** `{zscore_params['center']}`")
    st.sidebar.markdown(f"**Scale:** `{zscore_params['scale']}`")

    # Load and clean CSV
    df = pd.read_csv("Eva_csv - Sheet1.csv")
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Normalize country names
    country_norm = {
        "UK": "United Kingdom", "United Kingdom": "United Kingdom",
        "US": "USA", "USA": "USA", "United States": "USA",
        "United States of America": "USA",
        "South Korea": "South Korea", "Republic of Korea": "South Korea",
        "Cameroon": "Cameroon", "Canada": "Canada",
        "Kenya": "Kenya", "Pakistan": "Pakistan",
        "Switzerland": "Switzerland", "India": "India"
    }
    df['Country_clean'] = df['Country'].map(country_norm).fillna(df['Country'])

    # Build flag and display name
    def country_to_flag_manual(c):
        parts = str(c).split(",")
        return " ".join(flag_dict.get(country_norm.get(p.strip(), p.strip()), "") for p in parts)
    df['Flag'] = df['Country_clean'].apply(country_to_flag_manual)
    df['TeamDisplay'] = df.apply(lambda r: f"{r['Team name']} {r['Flag']}".strip(), axis=1)
    team_flag_map = df.set_index('Team name')['TeamDisplay'].to_dict()
        # Extraction function with sensitivity/error/late flagging
    def extract_metrics(row):
        # Parse submission date
        try:
            sub_dt = pd.to_datetime(row['Submission time'], format='%m/%d/%y %H:%M')
        except:
            sub_dt = None
        is_late = sub_dt is not None and sub_dt > pd.Timestamp("2025-07-19")
        raw = row['Evaluated_1']
        is_error = str(row.get('Flag','')).strip().lower() == 'error'

        # Handle explicit low-sensitivity message
        if isinstance(raw, str) and 'Sensitivity < 0.8' in raw:
            return {
                'Status':'Sensitivity < 0.8',
                'Submission time':row['Submission time'],
                'Team name':row['Team name'],
                'Affiliation':row.get('Affiliation',''),
                'Weighted Score':None,
                'Scaled Weighted Score':None,
                'AUPRC':None,
                'Net Benefit':None,
                'ECE':None,
                'F1':None,
                'Sensitivity':None,
                'Specificity':None,
                'Parsimony Score':None,
                'TP':None,'FP':None,'FN':None,'TN':None,
                'AUC':None
            }

        # Parse JSON metrics
        try:
            data = json.loads(raw)
            score = data.get('score', {})
            sens = score.get('Sensitivity')
            result = {
                'Submission time': row['Submission time'],
                'Team name': row['Team name'],
                'Affiliation': row.get('Affiliation',''),
                'Weighted Score': score.get('weighted_score'),
                'Scaled Weighted Score': score.get('scaled_weighted_score'),
                'AUPRC': score.get('AUPRC'),
                'Net Benefit': score.get('Net Benefit'),
                'ECE': score.get('ECE'),
                'F1': score.get('F1'),
                'Sensitivity': sens,
                'Specificity': score.get('Specificity'),
                'Parsimony Score': 1 - score.get('Parsimony Score', 0),
                'TP': score.get('tp'),
                'FP': score.get('fp'),
                'FN': score.get('fn'),
                'TN': score.get('tn'),
                'AUC': score.get('AUC')
            }
            # Assign status
            if is_error:
                result['Status'] = 'Error'
            elif is_late:
                result['Status'] = 'Late'
            else:
                result['Status'] = 'Complete' if (sens is not None and sens >= 0.80) else 'Low Sensitivity'
            return result
        except (json.JSONDecodeError, TypeError):
            return {
                'Status':'Error',
                'Submission time':row['Submission time'],
                'Team name':row['Team name'],
                'Affiliation':row.get('Affiliation','')
            }
        
    parsed = df.apply(extract_metrics, axis=1, result_type='expand')
    parsed['Country'] = df['Country_clean']
    parsed = parsed[parsed['Team name'].notna()]
    parsed['TeamID'] = parsed['Team name']
    parsed['Team name'] = parsed['Team name'].map(team_flag_map)

    # Round numeric metrics
    nums = parsed.select_dtypes(include='number').columns.difference(['Threshold Used','Inference Time'])
    parsed[nums] = parsed[nums].round(2)

    # Split into complete and flagged (Late, Error, Low Sensitivity)
    complete = parsed[parsed['Status']=='Complete']
    flagged = parsed[parsed['Status']!='Complete']

        # Prepare complete table
    complete_sorted = complete.sort_values('Scaled Weighted Score', ascending=False).reset_index(drop=True)
    complete_sorted.insert(0, 'Rank', complete_sorted.index + 1)
    # Only show up through AUC, then status and submission time
    cols = ['Rank', 'Team name', 'Affiliation', 'Weighted Score', 'Scaled Weighted Score', 'AUPRC', 'Net Benefit', 'ECE', 'F1', 'TP', 'FP', 'FN', 'TN', 'AUC']
    complete_sorted = complete_sorted[cols + ['Status', 'Submission time']]

    st.subheader('✅ Complete Submissions')
    st.dataframe(complete_sorted, use_container_width=True, hide_index=True, height=600)

    # Best performance per team (exclude Late)
    best_df = (
        parsed[parsed['Status']=='Complete']
              .sort_values('Scaled Weighted Score', ascending=False)
              .drop_duplicates(subset='TeamID')
              .reset_index(drop=True)
    )
    best_df.insert(0, 'Team Rank', best_df.index+1)
    best_cols = ['Team Rank','Team name','Affiliation','Scaled Weighted Score']
    st.subheader('🏆 Best Performance per Team')
    st.dataframe(best_df[best_cols], use_container_width=True, hide_index=True)

    with st.expander('⚠️ View Flagged/Error/Late Submissions'):
        flagged_sorted = flagged.sort_values('Scaled Weighted Score', ascending=False).reset_index(drop=True)
        #flagged_sorted.insert(0, 'Rank', flagged_sorted.index + 1)
        cols = ['Team name', 'Affiliation', 'Weighted Score', 'Scaled Weighted Score', 'AUPRC', 'Net Benefit', 'ECE', 'F1', 'TP', 'FP', 'FN', 'TN', 'AUC']
        flagged_sorted = flagged_sorted[cols + ['Status', 'Submission time']]
        st.dataframe(flagged_sorted, use_container_width=True, hide_index=True)

    #st.subheader("❌ Submissions with Errors (Flag/Error)")
    #st.dataframe(error_df.sort_values(by="Scaled Weighted Score", ascending=False),
    #            use_container_width=True, hide_index=True)
    


elif option == "Score Sensitivity Analysis":
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

    def compute_score_c(A, Ap, Nb, ECE, I, C, weights):
        return (weights['w_A'] * A) + \
               (weights['w_Ap'] * Ap) + \
               (weights['w_Nb'] * Nb) - \
               (weights['w_ECE'] * ECE) - \
               (weights['w_I'] * I) - \
               (weights['w_C'] * C)

    score_dynamic = compute_score_c(A=A_val, Ap=Ap_val, Nb=Nb_val, ECE=ECE_val, I=I_val, C=C_val, weights=dynamic_weights)

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
            (fixed_weights['w_Nb'] * Nb) - \
            (fixed_weights['w_ECE'] * ECE) - \
            (fixed_weights['w_I'] * I) - \
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
            (weights['w_Nb'] * Nb) - \
            (weights['w_ECE'] * ECE) - \
            (weights['w_I'] * I) - \
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
            (weights['w_Nb'] * Nb) - \
            (weights['w_ECE'] * ECE) - \
            (weights['w_I'] * I) - \
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

