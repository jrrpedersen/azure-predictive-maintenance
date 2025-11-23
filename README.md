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
├─ docs/
│  ├─ 01_infra_setup.md
│  ├─ 02_data_upload.md
│  ├─ 03_feature_engineering.md
│  ├─ 07_baseline_model_training.md
│  ├─ 09_cost_based_thresholding.md
│  ├─ 10_online_inference.md
│  └─ ...
├─ src/
│  └─ feature_engineering.py
├─ scripts/
│  ├─ infra/
│  │  ├─ register_datastore.py
│  │  ├─ upload_scania_data.py
│  │  └─ build_train_val_test_features.py
│  └─ ...
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

  * Data:
    - A multi-variate time series dataset collected collected from more than 33,000 SCANIA trucks.
    - The dataset includes operational data, truck specifications, and the repair records of an anonymized truck engine component called component X.
    - Publicly available and can be accessed on the Swedish National Data Service website (https://doi.org/10.5878/jvb5-d390)
    - Training dataset
      * train_operational_readouts.csv:
        - 1,122,452 observations or instances from 23,550 unique vehicles and 107 columns, including vehicle_id and time_step.
        - time_step acts as a gauge, measuring the duration in time_step that each vehicle has been utilizing Component X during its operational lifespan; vehicles do not necessarily follow the same sampling frequency in the time_step column.
        - 14 attributes are selected and anonymized in the operational data, offering a broad spectrum of information without divulging specifics about the nature of Component X. These variables are organized into single numerical counters and histograms where each histogram has several bins.
        - Six out of 14 variables are organized into six histograms with variable IDs: “167”, “272”, “291”, “158”, “459”, and “397,” with 10, 10, 11, 10, 20, and 36 bins, respectively.
        - The remaining eight are named “171_0”, “666_0”, “427_0”, “837_0”, “309_0”, “835_0”, “370_0”, “100_0”] are numerical counters.
        - The rate of missing values is low, with less than 1 percent missingness per feature/column.
      * train_tte.csv:
        - Contains the repair records of Component X collected from each vehicle, indicating the time_to_event (TTE), i.e., the replacement time for Component X during the study period
        - 23,550 rows
        - “length_of_study_time_step”: the number of operation time steps after Component X started working.
        - “in_study_repair”: the class label; set to 1 if Component X was repaired at the time equal to its corresponding length_of_study_time_step, or 0 in case no failure or repair event occurs during the first length_of_study_time_step of operation
        - Data is imbalanced with 21,278 occurrences of label 0 and 2,272 instances of label 1.
        - There are no missing values (shown by NaN) in this data file.
      * train_specifications.csv:
        - Contains information about the specifications of the vehicles, such as their engine type and wheel configuration.
        - There are 23,550 observations and eight categorical features, with no missing values for all vehicles.
        - The features can take categories in Cat0, Cat1, …, Cat28.
   - Validation dataset
     * validation_operational_readouts.csv:
       - Only a subset of the whole observations of each vehicle is provided, and it extends only up to a randomly selected readout.
       - Done to simulate the usage of a prediction model in a realistic scenario when it only has information about a vehicle up until the present time.
       - 196,227 observations/rows showing the number of instances from 5046 vehicles and includes 107 columns.
       - Contains a minimal missing value (less than one percent for each feature)
     * validation_labels.csv:
       - 5046 rows, equal to the number of vehicles contributed to the operational data of the validation set
       - class_label corresponds to the class for the last readout of each vehicle.
       - The temporal placement of this final simulated readout is categorized into five classes denoted by 0, 1, 2, 3, 4 where they are related to readouts within a time window of: (more than 48), (48 to 24), (24 to 12), (12 to 6), and (6 to 0) time_step before the failure, respectively.
       - This data set is also imbalanced and is skewed toward class 0, i.e., 4910 samples belong to class 0, while 76, 30, 16, and 14 samples belong to classes 4, 3, 1, and 2, respectively.
     * validation_specification.csv
       - No missing values
       - The data is collected from 5046 vehicles in the validation set.
   - Testing set
     * test_operational_readouts.csv
       - The last readouts of vehicles are randomly selected from the larger sequences observed during the study period
       - Contains 198'140 number of readouts from 14 variables (107 columns) gathered from 5045 unique vehicles
     * test_labels.csv:
       - The class label for 5045 vehicles in the test set that their last readouts are randomly selected in five classes of 0, 1, 2, 3, and 4.
       - This data file is also imbalanced and is skewed toward class 0, which contains 4903 samples. In comparison, classes 1, 2, 3, and 4 include 26, 15, 41, and 60 samples.
     * test_specifications.csv
       - 5045 test vehicles
       - Eight categorical features with values varying between Cat0, Cat1,..., and Cat28, with no missing values

* Sources
  * Azure Machine Learning documentation: https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2
  * Datasets:
   * SCANIA Component X dataset: A real-world multivariate time series dataset for predictive maintenance: https://www.nature.com/articles/s41597-025-04802-6?fromPaywallRec=false

