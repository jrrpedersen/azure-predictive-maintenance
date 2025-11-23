# Scania Predictive Maintenance on Azure

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Azure Machine Learning](https://img.shields.io/badge/Azure-ML-blueviolet.svg)](https://azure.microsoft.com/services/machine-learning/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Project-Portfolio-green.svg)](#)

End-to-end predictive maintenance project built on **Azure Machine Learning**, using the
public SCANIA truck dataset. The goal is to predict whether a vehicle’s Component X
will fail in the near future, and expose the model via an **online endpoint**.

The project demonstrates:

- Ingesting real-world PdM data into **Azure Data Lake Storage Gen2**
- Feature engineering from irregular time series → per-vehicle feature vectors
- Training & tuning **XGBoost** for rare-event classification
- Cost-based threshold selection for business-aware decisions
- Model registration in **Azure ML**
- Deployment as a **managed online endpoint** and calling it over HTTP

> ⚠️ This is a research / portfolio project. The current model improves over random
> but is **not** yet suitable for production maintenance decisions.

---

## Architecture overview

High-level architecture of the solution:

```text
           +-----------------------------+
           |     Local dev / GitHub      |
           |  - Notebooks & scripts      |
           |  - Feature engineering code |
           +--------------+--------------+
                          |
                          | (push code)
                          v
+-------------------------+--------------------------+
|                 Azure Machine Learning             |
|                                                    |
|  +--------------------+   +---------------------+  |
|  |  Compute Instance  |   |   Pipelines /       |  |
|  |  - Run notebooks   |   |   Training scripts  |  |
|  +---------+----------+   +----------+----------+  |
|            |                         |             |
|            | reads/writes            | writes      |
|            v                         v             |
|   +-------------------+   +---------------------+  |
|   |   ADLS Gen2       |   |  Model Registry     |  |
|   |  scaniapdmstorage |   |  XGBoost PdM model  |  |
|   +-------------------+   +----------+----------+  |
|                                        |           |
|                                        | deploy    |
|                                        v           |
|                            +-------------------+   |
|                            | Managed Endpoint  |   |
|                            |  /score           |   |
|                            +---------+---------+   |
+--------------------------------------+-------------+
                                         |
                                         | HTTP/JSON
                                         v
                               External clients / apps
```

Main components:

- **Azure Data Lake Storage Gen2 (`scaniapdmstorage`)** – stores the raw SCANIA CSV files and derived feature matrices.
- **Azure ML Workspace (`scania-pdm-ws`)** – central hub for experiments, compute and deployments.
- **Compute Instance** – runs notebooks and scripts for EDA, feature engineering, training, tuning and threshold search.
- **Model Registry** – holds the tuned XGBoost model (`scania-pdm-xgb-finetuned`).
- **Managed Online Endpoint** – hosts the scoring service that exposes `/score` for real-time predictions.

---

## Project structure

```text
.
├─ figures
│  └─ ...
├─ docs/
│  ├─ 01_setup_azure.md
│  ├─ 02_register_datastore.md
│  ├─ 03_upload_data.md
│  ├─ 04_register_data_assets.md
│  ├─ 05_eda_summary.md
│  ├─ 06_feature_engineering.md
│  ├─ 07_baseline_model_training.md
│  ├─ 08_model_tuning_results.md
│  ├─ 09_cost_based_thresholding.md
│  └─ 10_online_inference.md
├─ src/
│  └─ feature_engineering.py
├─ scripts/
│  ├─ infra/
|  |  ├─ connect_workspace_test.py
│  │  ├─ register_data_assets.py
│  │  ├─ register_datastore.py
│  │  └─ upload_scania_data.py
│  └─ build_train_val_test_features.py
├─ deployment/
│  ├─ score.py
│  ├─ conda.yaml
│  ├─ deploy_online_endpoint.py
│  └─ test_endpoint.py
├─ notebooks/
│  ├─ 01_exploratory_data_analysis.ipynb
│  ├─ 02_feature_engineering.ipynb
│  ├─ 03_model_training.ipynb
│  ├─ 04_model_tuning.ipynb
│  └─ 05_cost_based_threshold_search.ipynb
└─ README.md
```

---

## Data & feature pipeline (high level)

1. **Raw data**  
   - SCANIA dataset (train/validation/test CSVs)  
   - Uploaded to `scaniapdmstorage` in ADLS Gen2 under `scania-dataset/`

2. **Feature engineering**  
   - Irregular time-step sensor readouts aggregated per `vehicle_id`
   - Histogram channels → entropies, means, slopes, etc.
   - Counter channels → final values, trends, rates
   - Output:
     - `train_vehicle_features.csv`
     - `validation_vehicle_features.csv`
     - `test_vehicle_features.csv`

3. **Target definition**
   - Train: `in_study_repair` (0/1)
   - Validation/Test: `class_label` (0–4) collapsed to binary: `4 = failure`, others = non-failure

---

## Modeling summary

- **Model:** XGBoost classifier
- **Baseline metrics (test, threshold=0.5):**
  - ROC-AUC ≈ 0.70
  - PR-AUC ≈ 0.038
- **Tuned model:**
  - Hyperparameter search with `RandomizedSearchCV`
  - Test ROC-AUC ≈ 0.66
  - Test PR-AUC ≈ 0.045
- **Cost-based thresholding:**
  - Cost function: `Cost = 1 * FP + 50 * FN`
  - Optimal threshold on validation: **τ = 0.51**

More details in the `docs/` directory.

---

## Online endpoint

The tuned model is deployed as a **Managed Online Endpoint** in Azure ML.

Usage examples in:
- `deployment/test_endpoint.py`
- `docs/10_online_inference.md`
![Example of invoking the endpoint](/figures/azure_pdm_02_response_from_endpoint.png)

---

## How to reproduce

1. Set up infrastructure & upload data (`docs/01_infra_setup.md`, `docs/02_data_upload.md`)
2. Run feature engineering (`scripts/infra/build_train_val_test_features.py`)
3. Train & tune models (`notebooks/03_model_training.ipynb`, `04_model_tuning.ipynb`)
4. Threshold search (`05_cost_based_threshold_search.ipynb`)
5. Register & deploy model (`deployment/deploy_online_endpoint.py`)

---

## Limitations & future work

- Validation/test labels are derived from categorical buckets (0–4), adding noise.
- Precision remains low due to extreme class imbalance.
- Time-series models (LSTM/Transformers) not yet explored.
- Pipeline orchestration (Azure ML Pipelines) is a planned next step.

---

 ## Sources
* Azure Machine Learning documentation: https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2
* SCANIA Component X dataset: A real-world multivariate time series dataset for predictive maintenance: https://www.nature.com/articles/s41597-025-04802-6?fromPaywallRec=false

