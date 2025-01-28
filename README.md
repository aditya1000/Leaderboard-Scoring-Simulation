# 🏆 Leaderboard Score Sensitivity Analysis

Welcome to the **Leaderboard Score Sensitivity Analysis** tool for the **Pediatric Sepsis Data Challenge 2024**!

This tool allows you to adjust **Weights** and **Metrics** using the interactive controls in the sidebar. The main area dynamically displays the current leaderboard score and an interactive sensitivity analysis plot, which visualizes how changes in metrics impact the score.

---

## 🧮 Leaderboard Score Formula

### Components:
- **A**: Area Under the Receiver Operating Characteristic Curve (**AUC**)
- **Ap**: Area Under the Precision-Recall Curve (**AUPRC**)
- **Nb**: **Net Benefit**, measuring clinical utility
- **ECE**: **Expected Calibration Error** (Penalty for poorly calibrated probabilities)
- **I**: **Normalized Inference Time** (Penalty for high inference times)
- **C**: **Normalized Compute** (Penalty for high computational cost)

---

## 📊 Features

1. **Adjustable Weights**:
   - Configure weights (\(w_A, w_{Ap}, w_{Nb}, w_{ECE}, w_{I}, w_{C}\)) to explore their impact on the leaderboard score.

2. **Interactive Metrics**:
   - Modify the metrics (**A**, **Ap**, **Nb**, **ECE**, **I**, **C**) using sliders to reflect real-world predictions and costs.

3. **Sensitivity Analysis**:
   - Visualize how changes in a specific metric affect the leaderboard score using an intuitive plot.

4. **Dynamic Updates**:
   - Instant feedback on score changes based on the selected metric and weight adjustments.

---

## 🚀 How to Use

1. **Start the Tool**:
   - Run the Streamlit app:  
     ```bash
     streamlit run strmlit_LS3.py
     ```

2. **Adjust Weights**:
   - Use the **Weights** sliders in the sidebar to configure the importance of each metric or penalty.

3. **Adjust Metrics**:
   - Use the **Metrics** sliders to simulate performance values or penalties.

4. **View Sensitivity Analysis**:
   - Select a metric to vary and observe its impact on the leaderboard score in the interactive plot.

---

## 🎯 Goals

This tool helps participants of the **Pediatric Sepsis Data Challenge**:
- Understand the contribution of each metric to the leaderboard score.
- Optimize their models by balancing performance and computational efficiency.
- Gain insights into the sensitivity of the scoring system.

---

## 🛠 Technical Requirements

- **Python 3.8+**
- **Streamlit 1.25.0**
- **Altair 5.0.1**
- **Pandas 1.5.4**
- **NumPy 1.23.5**

Install dependencies:
```bash
pip install streamlit altair pandas numpy
