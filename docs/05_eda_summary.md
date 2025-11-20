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
