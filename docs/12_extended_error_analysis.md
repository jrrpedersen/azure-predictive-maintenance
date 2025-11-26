
# 11 - Extended Error Analysis  
**Predictive Maintenance Model – False Positives, False Negatives & Failure Mechanisms**  
*This document extends and deepens the analyses from `11_error_analysis.md`.*

---

# 1. Introduction  
The initial error analysis revealed several patterns in the model’s false positives (FP) and false negatives (FN), including cluster structure, suspicious dependence on `study_length_time_step`, and interactions between sensor-based time‑series features.  

This extended analysis adds five major components:

1. SHAP summary comparisons for FP vs FN  
2. Partial Dependence + ICE behavior of key features  
3. Feature interaction analysis  
4. Mutual Information ranking  
5. Reliability diagrams  
6. Hypothesis tests (FP vs FN distributions)  
7. Error archetypes  
8. Recommendations and future work  

Together, these analyses provide a multi‑angle understanding of when and why the model fails, revealing both physical patterns and dataset artifacts.

---

# 2. SHAP Summary Plots – FP vs FN Comparison  

We computed SHAP values for the two error groups using a *proper* `model.predict`‑based SHAP explainer to ensure feature alignment. The figures show the **20 most influential features for each group**.

### Key Insights  

## 2.1 For False Positives  
FP predictions are strongly driven by:

- **High slopes** (e.g., `835_0_slope`, `427_0_slope`)  
- **High mean or max sensor values** (`397_0_mean`, `397_3_max`, `158_9_max`)  
- **High variability features** (centroid and STD features)  
- Moderate values of `study_length_time_step`, but less extreme compared to FN

**Interpretation**:  
The model sees patterns that *look like early degradation* (steep slopes, rising means) and predicts failure even when the vehicle does *not* fail. These are cases of:  

- transient anomalies  
- noisy sensors  
- high-slope events that do not lead to failures  

---

## 2.2 For False Negatives  
FN predictions are associated with:

- **Very short `study_length_time_step` values**  
- **Only mild slopes and moderate sensor readings**  
- Less pronounced gradients in the key failure‑related channels  
- Sensor values that look "normal" even though failure occurs shortly

**Interpretation**:  
False negatives usually occur when the vehicle enters the observation window *late*-too little history exists to show degradation. This suggests a **sampling or labeling bias**:  
the failure label appears without sufficient precursor data.

---

# 3. Partial Dependence & ICE (Individual Conditional Expectation)

Partial Dependence (PD) combined with ICE gives insight into how the model responds when varying one feature while keeping all others fixed.

### Features examined:
- `study_length_time_step`  
- `291_centroid_std`  
- `427_0_slope`  
- `397_0_mean`  
- `397_3_max`  
- `171_0_r2`  
- `835_0_slope`

### 3.1 Most important PD/ICE findings

## (a) `study_length_time_step`
The model’s output drops sharply as soon as study length increases.  
This indicates a **strong global prior effect**:

> Vehicles with short study length are “seen” as failure‑prone.

This is likely **dataset-based** rather than a physical effect.

---

## (b) `291_centroid_std`
Shows moderate increases in failure probability when variability is high, but the effect is weaker than slopes and means.

---

## (c) Slopes and Means (e.g., `835_0_slope`, `397_0_mean`, `397_3_max`)
These show:

- clear monotonic increases in failure probability  
- strong agreement with physical degradation phenomena  
- FN cases typically lie in low-slope regions  
- FP cases lie in high-slope, high‑mean regions  

This validates that the model *correctly learned real physical degradation patterns*, but occasionally overreacts.

---

# 4. Feature Interactions (2D PD)

Two interaction plots were generated:

1. `study_length_time_step × 291_centroid_std`  
2. `835_0_slope × 397_0_mean`

### 4.1 Key findings

## (a) Time Length × Variability  
`study_length_time_step` dominates the prediction, with the interaction showing that **shorter study windows always increase failure risk**, no matter what `291_centroid_std` is.

This is **not physically meaningful** → suggests dataset bias.

## (b) Slope × Mean  
The interaction between `835_0_slope` and `397_0_mean` is physically meaningful:

- High slope **and** high mean = highest risk  
- Moderate values produce moderate risk  
- Low slope & low mean = safe region  

This explains clusters of FP that have the slope/mean signature of degradation but never fail.

---

# 5. Mutual Information Ranking – FP vs FN Separation

We computed MI scores measuring ability of each feature to separate FN from FP.

### Top features:
- `study_length_time_step` (highest MI)  
- Slope features (e.g., `158_3_mean`, `427_0_delta`, `171_0_delta`)  
- Variability features (`total_std`, `centroid_std`)  
- Max features (`459_15_max`, `158_9_max`)

### Interpretation
- **MI confirms the strong role of `study_length_time_step`** in distinguishing FPs from FNs.  
- FN = short study length  
- FP = normal study length + strong degradation signals  

This reinforces earlier conclusions.

---

# 6. Reliability Diagrams

Two reliability diagrams were computed:

- Global  
- Mid-probability region (0.2–0.8)

### Observations

- The model **overestimates** the failure probability across almost all bins.  
- The true fraction of positives is extremely low → points cluster near 0.  
- Brier scores are above 0.1, showing **poor calibration**.

### Interpretation
This model is **useful for ranking risk** but **not for calibrated probability estimation**.

Failure is very rare → calibration is inherently difficult, and rebalancing or Platt scaling / isotonic regression may be useful.

---

# 7. Hypothesis Tests (FP vs FN)  
We ran Mann–Whitney U-tests comparing FP vs FN distributions for top features.

### Findings
- Many features show extremely low p-values (< 1e‑12) after FDR correction.  
- Strongest separation occurs for slope and delta features (e.g., `427_0_mean`, `171_0_mean`, `171_0_delta`).  
- `study_length_time_step` is also highly significant.

### Interpretation
FP and FN groups come from **statistically different regions of feature space**.  
This confirms the cluster analysis and UMAP structure:

- **FN** = incomplete data history  
- **FP** = strong-but-not-fatal anomalies

---

# 8. Error Archetypes  
We computed mean values for FP, FN, and Correct samples for 5 representative features.  

### Findings

- FN have **short study length**, much higher `158_9_max`.
- FP have **high slopes** but *lower* `397_0_max` than true failures.
- Correct samples sit between FP and FN except for `study_length_time_step`.

### Interpretation
FP and FN are *not symmetric mistakes*.  
They represent **different physical or procedural failure modes**.

---

# 9. Summary of Findings

## 9.1 What the model learned correctly
- Slopes and means of key sensors are powerful indicators of degradation.  
- Interactions between slope and mean features reflect real-world behavior.  
- High variability in certain channels correlates with increased failure likelihood.

## 9.2 What the model learned incorrectly (data artifacts)
- Strong, unintended reliance on `study_length_time_step`.  
- FN often have very incomplete time history.  
- FP often include transient anomalies that appear failure-like.

## 9.3 Why errors occur
- **False Negatives**: Insufficient precursor history (labeling artifact).  
- **False Positives**: Non-fatal anomalous behavior that mimics degradation patterns.

---

# 10. Recommendations & Future Work

## 10.1 Remove or reformulate `study_length_time_step`
This feature is leaking workflow or sampling information into the model.  
Consider:

- normalizing by total available time  
- using relative position in lifecycle  
- removing it entirely

---

## 10.2 Use time-series representations rather than aggregated features

The current approach uses aggregated statistics (mean, slope, max), which lose temporal structure. Consider:

- 1D CNNs  
- LSTMs/GRUs  
- Temporal Convolutional Networks  
- Transformers for time series  

This would reduce reliance on proxy variables.

---

## 10.3 Add context features  
Metadata may help disambiguate FPs:

- vehicle type  
- usage patterns  
- operating conditions  

---

## 10.4 Calibrate the model  
Apply:

- Platt scaling  
- Isotonic regression  
- Conformal prediction  

to generate **trustworthy** probability estimates.

---

## 10.5 Stratified evaluation of sensor stability  
Some FP behaviors may come from unstable channels.  
Consider:

- sensor health indicators  
- variance-based filtering  
- cross-channel consistency checks  

---

## 10.6 Improve negative labeling  
FN often arise because the failure appears suddenly in the dataset.  
Future labeling processes could include:

- expanding the precursor window  
- ensuring uniform observation history  
- rebalancing samples with poor temporal coverage

---

# 11. Final Conclusion  
The extended analysis confirms and refines earlier insights:

- The model captures real degradation physics through slope/mean interactions.  
- It also learns sampling artifacts, especially `study_length_time_step`.  
- FN happen when the model lacks temporal context; FP happen when anomalies mimic failure.  
- Calibration is poor due to extreme class imbalance.  
- Future models should reduce reliance on proxy variables and incorporate richer temporal features.

This document provides a complete understanding of how the model behaves, why misclassifications occur, and how to improve future predictive maintenance models.

