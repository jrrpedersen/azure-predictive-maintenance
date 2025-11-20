# SCANIA Predictive Maintenance — EDA Summary
## Overview
This document summarizes the first-stage Exploratory Data Analysis (EDA) performed on the SCANIA Component X Predictive Maintenance dataset, stored in Azure Data Lake Storage and accessed through Azure ML Studio.
The goal of this EDA is to:
* Understand the dataset composition
* Inspect missing values and data quality
* Explore class imbalance and target distribution
* Examine the temporal structure of vehicle operational data
* Prepare insights that guide our modeling and feature engineering strategy
The findings here inform the next stage: data engineering and model pipeline design.

## 1. Dataset Structure
The training dataset consists of three CSV files:
### 1. Operational Readouts (1.12M rows, 107 columns)
Contains time-step sensor data for each vehicle.
* Shape: 1,122,452 rows × 107 columns
* Columns include:
  - vehicle_id
  - time_step
  - 8 numerical counters
  - 6 histogram-based features with multiple bins (e.g., 167_0 to 167_9, 291_0 to 291_10, etc.)

### 2. TTE (Time to Event / Label Table) (23,550 rows)
One row per vehicle.
Columns:
* vehicle_id
* length_of_study_time_step
* in_study_repair (target variable; 1 = failure)

### 3. Specifications (23,550 rows)
Static categorical metadata for each vehicle.
Columns:
* vehicle_id
* Spec_0 to Spec_7 (all categorical strings)
* All vehicles in TTE/specifications have corresponding operational readouts.

## 2. Missing Values & Data Quality
### Operational Readouts
There are small but notable pockets of missing values:
* Histogram feature 291_x bins: ~0.86% missing
* Histogram feature 459_x bins: ~0.74% missing
All other sensor columns: no missing values.
Given the large row count, this represents thousands of missing entries.

### TTE
No missing values.

### Specifications
No missing values.

### Engineering Impact
* Missingness is low enough to handle easily.
* Median imputation per feature is the likely first choice.
* More advanced approaches (per-vehicle interpolation) remain possible.

## 3. Target Variable Analysis
| Value       | Count  | Proportion |
| ----------- | ------ | ---------- |
| 0 (healthy) | 21,278 | 90.35%     |
| 1 (failed)  | 2,272  | 9.65%      |

### Engineering Impact
The dataset is clearly imbalanced, typical of predictive maintenance.
Modeling will require:
* Class weighting (e.g., scale_pos_weight in XGBoost) or
* Oversampling (SMOTE) or
* Threshold tuning
* Metrics like ROC-AUC, PR-AUC, precision-at-k
Accuracy alone will be misleading here.

## 4. Temporal Structure - Sequence Lengths Per Vehicle
Each vehicle has a sequence of time-step sensor readings.

### Sequence length per vehicle (summary):
* Mean: 47.7 time steps
* Std: 27.4
* Min: 5
* Median: 43
* 75th percentile: 64
* Max: 303

### Key observations
* Vehicles have highly variable sequence lengths
* time_step is relative per vehicle, not globally aligned across vehicles
* Sampling rate is not guaranteed to be consistent
* There is a long tail of vehicles with >200 time steps

### Engineering Impact
This has major implications:
1. Traditional models (XGBoost, RandomForest) require fixed-size feature vectors.
→ We must aggregate each vehicle’s sequence into summary statistics.
2. Sequence models (LSTM, Transformers) require:
  * padding & masking
  * caution due to uneven sampling
  * more complex engineering
Thus, aggregation-based modeling will be our first approach.

## 5. Specifications Table - Categorical Metadata
Eight categorical fields per vehicle:
* Spec_0 ... Spec_7
* All are object dtype
* No missing values
* Likely represent:
  - model version
  - configuration
  - production metadata

### Engineering Impact
These features require encoding:
* One-hot encoding (safe baseline)
* Target encoding (for tree models)
* Embeddings (if using deep learning)

## 6. Operational Variables - Sensor Types
There are two major sensor groups:

### 1. Numerical counters (e.g., 171_0, 666_0, etc.)
Single continuous values per time_step.

### 2. Histogram-based variables
Each histogram is represented as multiple bins:
Examples:
* 167_0 to 167_9
* 291_0 to 291_10
* 459_0 to 459_19
* 397_0 to 397_35
These are sensor distributions captured at each time_step.

### Engineering Impact
Histogram variables will require careful aggregation:
* Mean across time
* Variance
* Dominant bin
* Entropy
* Slope of histogram centroid
* Range or percentiles
These are high-value signals for degradation modelling.

## 7. Preliminary Modeling Strategy
Based on the EDA, a robust first modeling pipeline should:

### ✔ Use vehicle-level aggregation
(one row per vehicle)

### ✔ Include:
* aggregated histogram statistics

* aggregated numerical counters

* encoded specifications

### ✔ Handle class imbalance using:
* class_weight
* or XGBoost’s scale_pos_weight

### ✔ Use train/validation/test splits provided by the dataset
(no leakage risk)

### ✔ Candidate first models:
* XGBoost (strong baseline)
* LightGBM
* Logistic regression (sanity baseline)

### ✔ Future extension:
Sequence modeling (LSTM/Transformer) once baseline is completed.

## 8. Summary of Key Insights
*  The dataset is large and rich (1.1M+ rows of sensor data).
*  Missing values exist but are small and manageable.
*  The target is imbalanced (≈10% failures).
*  Time-step sequences per vehicle are highly variable.
*  Histogram sensor features require thoughtful aggregation.
*  Specifications are categorical and must be encoded.
*  A vehicle-level aggregated model is the most appropriate starting point.

This EDA gives us a solid understanding of the dataset and provides a clear direction for the next stage.
