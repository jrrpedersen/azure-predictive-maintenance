# 09 - Cost-Based Threshold Optimization

This document describes the experiment performed to optimize the classification threshold of the tuned XGBoost model, using a cost-sensitive objective that reflects real-world predictive maintenance priorities.

The goal is to reduce the operational cost of predictions, where missing a true failure (FN) is very expensive, while a false alarm (FP) is relatively cheap.

---

## 1. Motivation

The tuned XGBoost model showed:

- Acceptable ROC-AUC (≈0.66-0.70)
- Very low PR-AUC (≈0.04-0.05)
- Poor precision on the minority class
- Moderate recall at high cost
- Considerable gap between TRAIN and VAL/TEST performance

This is typical in rare-event classification and predictive maintenance, where true failures are extremely scarce and difficult to distinguish from normal operation.

Since false negatives are far more costly than false positives, threshold optimization is a natural next step.

---

## 2. Methodology

We performed cost-based threshold search using the tuned model probabilities.

### Cost Function

```python
Cost = FP * 1 + FN * 50
```

Where:

- **FN cost (50)** represents the high cost of missing a failure  
- **FP cost (1)** represents the low cost of unnecessary maintenance  

### Procedure

1. Load tuned model (`xgb_pdm_finetuned.pkl`)
2. Predict failure probabilities on validation set
3. Sweep threshold values from **0.01 to 0.99**
4. For each threshold:
   - Compute FP and FN  
   - Compute total cost  
5. Select threshold τ minimizing validation cost
6. Recompute performance metrics (ROC-AUC, PR-AUC, Precision, Recall, F1, confusion matrix)
7. Evaluate same threshold on test data

This simulates real-world cost-aware maintenance decision-making.

---

## 3. Results

### **Optimal cost-based threshold**

```
τ = 0.51
```

This threshold was applied to both validation and test splits.

---

## 4. Validation Results @ τ = 0.51

```
=== VALIDATION (TUNED, COST-OPTIMAL) @ threshold = 0.510 ===
ROC-AUC:   0.6669
PR-AUC :   0.0417
Precision: 0.0400
Recall:    0.2895
F1-score:  0.0703
Cost:      3228.0
```

### Confusion Matrix

```
[[4442  528]]    FP = 528
[[  54   22]]    FN = 54
```

### Interpretation

- Recall improves sharply to **28.9%**
- Precision remains extremely low (**4%**)
- Many positives still missed, many false alarms generated
- Cost remains high due to many FP and FN

---

## 5. Test Results @ τ = 0.51

```
=== TEST (TUNED, COST-OPTIMAL) @ threshold = 0.510 ===
ROC-AUC:   0.6597
PR-AUC :   0.0451
Precision: 0.0251
Recall:    0.3500
F1-score:  0.0468
Cost:      2766.0
```

### Confusion Matrix

```
[[4169  816]]    FP = 816
[[  39   21]]    FN = 39
```

### Interpretation

- Recall rises to **35%**
- Precision drops to **2.5%** (very poor)
- Model requires many false alarms to detect few true failures
- Threshold tuning helps but does not solve separability issues

---

## 6. Key Takeaways

### 1. Threshold tuning boosts recall but not separability

Even with aggressive cost weighting (50:1):

- Many false alarms  
- Many missed failures  
- Weak probability calibration  
- Limited discriminatory power  

### 2. PR-AUC exposes the real difficulty

**Test PR-AUC:**  
- Baseline: 0.0377  
- Tuned: **0.0451**

Improved, but still extremely low → model struggles in rare-event detection.

### 3. ROC-AUC is misleading under extreme imbalance

ROC curves hide poor precision; PR-AUC is the correct metric.

### 4. Strong overfitting persists

- TRAIN AUC ≈ **0.999**
- VAL/TEST AUC ≈ **0.66–0.70**

Model memorizes noise instead of generalizing useful patterns.

---

## 7. Implications for Production Systems

A production PM model must:

- Trigger alarms with useful precision  
- Maintain adequate recall  
- Provide calibrated risk scores  

The current model:

- Useful for **fleet-level** prioritization  
- Not suitable for **vehicle-level** predictions  

Operational risks:

- Excessive false alarms → unnecessary maintenance  
- Missed failures → breakdown and safety risk  

---

## 8. Next Steps

### A. Better Feature Engineering (Most Important)

- Time-series trend extraction  
- Rolling window stats  
- Degradation indicators  
- Change-point detection  
- Frequency-domain features  
- Time-to-event weighted features  

### B. Correct Label Construction

Validation/test labels use a coarse 0–4 proximity score.  
Mapping only “4” → failure oversimplifies real degradation dynamics.

### C. Alternative Modeling Approaches

- Balanced gradient boosting  
- Ranking models (`rank:pairwise`)  
- Survival models (Cox, DeepSurv)  
- Temporal models (LSTM, GRU, TCN)  
- Autoencoders/anomaly detection  

### D. Probability Calibration

Improves threshold stability and interpretability.

---

## 9. Summary

Cost-based threshold optimization allowed the model to:

- Improve recall  
- Reduce operational cost relative to naive thresholds  
- Expose the value of explicit cost-sensitive evaluation  

However:

- Precision remains too low for deployment  
- PR-AUC indicates limited predictive power  
- Overfitting persists  
- Feature engineering and labeling need improvement  

This concludes the tuning phase and motivates improved features, labeling, and advanced modeling in the next development stage.

