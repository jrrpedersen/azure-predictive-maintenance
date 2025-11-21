# 08 Baseline Model Training

This document summarizes the development, evaluation, and registration
of the baseline predictive maintenance model for the Scania dataset.

The goal of the baseline model is not to achieve production-grade
performance, but to establish:

-   A fully reproducible end-to-end workflow\
-   A defensible benchmark for future model improvements\
-   A model registered in Azure ML's Model Registry\
-   A solid foundation for subsequent hyperparameter tuning, class
    rebalancing, and enhanced feature engineering

## 1. Objective

The task is to predict whether a vehicle is close to failure, defined
as:

    class_label = 4 → imminent failure within approximately 0–6 timesteps

Because the training labels (`train_tte.csv`) provide a binary target
(`in_study_repair ∈ {0,1}`), while validation/test labels provide
5-class *distance-to-failure*, we converted:

-   `in_study_repair == 1` → **failure**\
-   `class_label == 4` → **failure**\
-   all other classes → **non-failure**

This yielded a **binary classification problem** consistent across
train/val/test sets.

## 2. Feature Engineering Summary

Feature engineering was performed using:

-   Operational sensor histories (107 sensor channels, \~1.1M rows)\
-   Vehicle specifications (categorical)\
-   Failure information from TTE (train) or `class_label` (val/test)

All engineered features are **per-vehicle aggregated statistical
summaries**.

The full feature set includes:

### Basic statistics (per sensor)

-   `mean`, `std`, `min`, `max`, `median`\
-   quantiles: `10%`, `25%`, `75%`, `90%`

### Histogram-based features (per sensor group)

For grouped channels (e.g., `167_X`, `291_X`, `459_X`, etc.):

-   histogram bin means\
-   histogram bin standard deviations\
-   histogram entropy

### Trend features (for counter-like sensors)

-   linear regression slope over time\
-   `R²` of regression fit\
-   final observed value\
-   rate of increase

### Categorical encodings

-   One-hot encoding for `Spec_0 – Spec_7`

After all transformations, the final feature matrix contained **577
features per vehicle**.

Generated feature files:

-   `train_vehicle_features.csv`\
-   `validation_vehicle_features.csv`\
-   `test_vehicle_features.csv`

## 3. Baseline Model

A standard **XGBoost binary classifier** was trained using default-ish
parameters:

``` python
xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1
)
```

No class weighting, hyperparameter tuning, or resampling was used.

## 4. Model Performance

### 4.1 TRAIN Performance

  Metric    Value
  --------- --------
  ROC-AUC   0.9884
  PR-AUC    0.9043

**Confusion Matrix (threshold = 0.5)**

    [[19436  1842]
     [   34  2238]]

### 4.2 VALIDATION Performance

  Metric    Value
  --------- --------
  ROC-AUC   0.6997
  PR-AUC    0.0490

**Confusion Matrix (threshold = 0.5)**

    [[3765 1205]
     [  40   36]]

### 4.3 TEST Performance

  Metric    Value
  --------- --------
  ROC-AUC   0.7020
  PR-AUC    0.0377

**Confusion Matrix (threshold = 0.5)**

    [[3445 1540]
     [  26   34]]

## 5. Model Registration

``` python
model = Model(
    name="scania-pdm-xgb-baseline",
    path="xgb_baseline_model.pkl",
    type="custom_model",
    description="Baseline XGBoost model for Scania predictive maintenance",
)
ml_client.models.create_or_update(model)
```

## 6. Discussion

### Strengths

-   Fully reproducible workflow\
-   Strong training performance\
-   Good failure recall on val/test\
-   Reasonable benchmark

### Weaknesses

-   Severe class imbalance\
-   PR-AUC near zero\
-   Overfitting\
-   Threshold 0.5 inappropriate\
-   No hyperparameter optimization

## 7. Next Steps

### Improve labels

-   Revisit definition of failure

### Apply class imbalance strategies

-   `scale_pos_weight`\
-   Balanced weights\
-   Oversampling / undersampling

### Hyperparameter tuning

-   Azure ML sweeps\
-   Threshold optimization\
-   PR-AUC objective

### Feature engineering improvements

-   Temporal features\
-   Segmented trends\
-   Advanced histogram stats\
-   Missing-value encoding

### Deployment

-   Register improved versions\
-   Deploy scoring endpoint\
-   Inference pipeline

## 8. Conclusion

The baseline model represents a complete, reproducible first iteration
of the predictive maintenance pipeline.\
It provides a solid benchmark for future modeling and is now registered
in Azure ML for use in downstream tasks.
