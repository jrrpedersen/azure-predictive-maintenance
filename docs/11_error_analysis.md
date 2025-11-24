
# 11. Error Analysis

This document presents a comprehensive error analysis performed as part of the **Scania Predictive Maintenance on Azure** portfolio project.  
The goal of this analysis is to understand **why the trained XGBoost model makes false positives (FP)** and **false negatives (FN)** when predicting component failures, and to extract actionable insights that can guide future improvements.

The analysis integrates:
- Exploratory statistics on FP/FN groups  
- Feature-level contrasts using the model’s top 20 most important features  
- Dimensionality reduction (UMAP) to understand structure  
- Clustering (KMeans) of FP samples to identify behavioral regimes  
- SHAP explanations of individual misclassified samples  
- Outlier detection  
- Summary conclusions and recommendations  

---

## 1. Overview of the Error Analysis Approach

After completing feature engineering and model training, we constructed a unified dataset:

```python
df_err = X_test.copy()
df_err["y_true"] = y_true
df_err["y_proba"] = y_proba
df_err["y_pred"] = y_pred
df_err["error_type"] = ["Correct", "FP", "FN"]
```

This enabled analysis from three complementary directions:

1. **Probability-space analysis**  
   - Distribution of predicted probabilities for FP/FN/Correct  
   - Identifying borderline vs extreme misclassifications  

2. **Feature-space analysis**  
   - PCA / UMAP projections  
   - Cluster structure among false positives  
   - Mean-value contrasts across FP vs FN vs Correct  

3. **SHAP-based interpretability**  
   - Understanding which features drive FP and FN cases  
   - Verifying whether specific sensors or temporal patterns cause errors  

4. **Statistical contrasts**  
   - Cohen’s *d* effect sizes between FP and FN  
   - Identifying the most discriminative features  

Together these provide a deep and multi-angle view of **why the model fails**.

---

## 2. Probability Distributions by Error Type

Plotting predicted probabilities (`y_proba`) by error type revealed three clear behaviors:

### **False positives (FP)**  
Cluster around mid-to-high probability (0.55–0.9).  
These trucks were *not* failures but looked deterioration-like to the model.

### **False negatives (FN)**  
Cluster around very low predicted probabilities (0.0–0.15).  
Even though they *were* failures, their sensor profiles overlapped with normal operating variation.

### **Correct predictions**  
Span the full range but concentrate near extremes (0.0 for healthy, 0.9+ for failures).

**Conclusion:**  
False positives occur when the model sees “incipient failure patterns” in healthy trucks.  
False negatives occur when true failures resemble normal operating conditions.

---

## 3. UMAP Projection and Error Geography

UMAP embeddings revealed a **clear global structure** in the engineered feature space:

- Most normal (Correct) samples form a broad manifold  
- FP samples lie in two distinct arcs  
- FN samples lie scattered but concentrated near normal regions  

### **Important observation:**  
Within the lower arc, FPs tend to accumulate toward **higher UMAP1 values**, reflecting feature patterns related to elevated or trending sensor signals.

---

## 4. Clustering False Positives (KMeans)

Applying KMeans (k=3) on FP-only samples revealed **three consistent FP modes**:

### **Cluster 1 – “Sustained High Baseline” FPs**
- Elevated centroids  
- High minimum values  
- Elevated histogram bins  
- Corresponds to the right arc of UMAP  

These trucks operate with **consistently high sensor baselines**, which mimic degradation signatures.

### **Cluster 0 – “Spiky / Noisy Sensors” FPs**
- High deltas  
- High variance  
- Occasional extreme spikes  

These profiles resemble *abrupt failure patterns* even though the trucks are healthy.

### **Cluster 2 – Rare Outliers**
- Very unusual sensor distributions  
- Possibly uncommon operating modes or mislabeled samples  

---

## 5. Feature-Level Statistical Contrast (Top 20 Features)

Using the top 20 most important XGBoost features, we compared mean values across FP/FN/Correct.

### General patterns:

| Feature Type | FP | FN | Correct |
|--------------|----|----|---------|
| Max values | Low | Very high | High |
| Mean values | Low/moderate | Very high | High |
| Trend/slope | Moderate | Highest | Low |
| Histogram categories | Varies | Some bins never triggered | Moderate |
| End-of-sequence values | Moderate | Highest | Moderate |

### Key insights:

- **FNs have the strongest failure signatures** (very high maxima, means, slopes).  
- **FPs do *not* look like failures in magnitude**, but exhibit suspicious *shapes* (rising trends, elevated centroids).  
- **Correct cases sit between FP and FN**, showing that many failure signatures overlap with normal operation.

This confirms the model’s sensitivity to **shape-based features**, particularly *slopes and last values*.

---

## 6. Cohen’s d (Effect Size) Between FP and FN

To quantify FP–FN separation:

\[
d = \frac{\mu_{\mathrm{FP}} - \mu_{\mathrm{FN}}}{\sigma_{\mathrm{pooled}}}
\]

Consistently:

- Features with highest effect sizes: **slopes, centroids, and histogram bins**  
- FN dominance in magnitude-based features  
- FP dominance in shape-based and distributional features  

This numerically confirms the qualitative findings.

---

## 7. SHAP Explanations

SHAP waterfall plots consistently showed:

- **171_0_last** → strong positive contribution toward predicting failure  
- **171_0_first** → strong negative contribution  
- Their difference encodes a **trend / slope** pattern  

For both FP and FN samples, SHAP highlights:

### False positives  
Triggered by *rising end values* or subtle multi-sensor centroid shifts.

### False negatives  
Although high magnitude readings exist, the slope-based features insufficiently capture deterioration trajectory.

### Model weakness revealed:  
It overly relies on **trend indicators**, underweights absolute-level indicators.

---

## 8. Outlier Detection

Using Isolation Forest:

- FP: **5 outliers**  
- FN: **0 outliers**  
- Most outliers occur in the correct class  

This confirms:

- FP clustering was *not* due to random noise  
- FN samples tend to be “inliers” that overlap with normal regions  
- The FP clusters truly represent systematic *operational regimes*  

---

## 9. Summary of Findings

### ✔ FP arise from two main patterns:
1. **Sustained elevated baselines**  
2. **Spiky or noisy sensors with rising slopes**

### ✔ FN arise when:
- True failure signals overlap with normal variation  
- Failure trajectories are gradual, not sharply trending  
- Sequence length is long, creating diluted slopes  

### ✔ Model over-relies on:
- Trend/slope indicators  
- End-of-sequence values  
- Certain histogram categories  

### ✔ Misses signals in:
- Absolute magnitudes  
- High global sensor levels  
- Slow deterioration patterns  

---

## 10. Recommendations for Future Improvements

### **A. Include absolute-level features explicitly**
Normalize slope-based features relative to overall magnitude.

### **B. Train cluster-aware models**
Separate models for:
- high-baseline regime  
- spike-heavy regime  

### **C. Add temporal models**
Transformers / LSTMs could capture deterioration trajectories directly.

### **D. Rebalance loss to reduce FN**
Context-dependent cost functions (already used) could be expanded.

### **E. Smooth sensor signals**
Reduce sensitivity to random spikes.

---

## 11. Conclusion

This error analysis revealed that the model’s misclassifications are **not random**, but arise from identifiable sensor behavior regimes.  
By combining UMAP, clustering, statistical contrasts, and SHAP explanations, we uncovered how certain temporal patterns and operating modes drive both false positives and false negatives.

These findings directly inform how the model and feature engineering pipeline should evolve in future iterations of the predictive maintenance system.

