## Overview

This document describes how the SCANIA predictive maintenance dataset is transformed from raw time-step data into a **per-vehicle feature matrix** suitable for machine learning.

The feature engineering logic is split into two layers:

1. A **reusable Python module**:  
   `src/feature_engineering.py`
2. A **driver script** that builds all splits (train/validation/test):  
   `scripts/build_train_val_test_features.py`

This design allows:

- interactive exploration and debugging in notebooks, and  
- reproducible batch generation of features for all data splits,  
- future reuse in Azure ML pipelines.

---

## 1. Inputs and Outputs

### 1.1 Raw inputs (in ADLS Gen2)

From the Azure Data Lake Storage Gen2 account `scaniapdmstorage`, filesystem `scania-dataset`, we use:

- **Training:**
  - `train/train_operational_readouts.csv`
  - `train/train_tte.csv`
  - `train/train_specifications.csv`

- **Validation:**
  - `validation/validation_operational_readouts.csv`
  - `validation/validation_labels.csv`
  - `validation/validation_specifications.csv`

- **Test:**
  - `test/test_operational_readouts.csv`
  - `test/test_labels.csv`
  - `test/test_specifications.csv`

### 1.2 Engineered outputs (per-vehicle features)

The feature script produces three CSV files (stored initially on the local filesystem of the environment where the script is run):

- `train_vehicle_features.csv`
- `validation_vehicle_features.csv`
- `test_vehicle_features.csv`

Each file contains:

- one row per `vehicle_id`
- hundreds of engineered features
- the target label: `in_study_repair` (0/1)

These files are **derived artifacts** and can be regenerated from raw data, so they are not committed to GitHub.

---

## 2. Feature Engineering Logic (`src/feature_engineering.py`)

The module `src/feature_engineering.py` encapsulates the feature engineering logic used for all splits.

Key components:

### 2.1 Merging raw tables

For each split (train/val/test), three raw tables are merged:

1. **Operational readouts**
   - time-step level sensor data
   - columns:
     - `vehicle_id`
     - `time_step`
     - counter variables
     - histogram-bin variables

2. **TTE (Time to Event)**
   - one row per vehicle
   - columns:
     - `vehicle_id`
     - `length_of_study_time_step`
     - `in_study_repair` (target label)

3. **Specifications**
   - static metadata per vehicle
   - columns:
     - `vehicle_id`
     - `Spec_0` … `Spec_7` (categorical)

The merged dataset is:

- time-step level,
- with sensor data, label, and specifications aligned by `vehicle_id`.

---

### 2.2 Counter-based features

A set of columns are treated as *counters*, for example:

- `171_0`, `666_0`, `427_0`, `837_0`,
- `309_0`, `835_0`, `370_0`, `100_0`

For each vehicle and each counter column, the module computes:

- `first` value in time
- `last` value in time
- `delta` = last – first
- `mean`
- `std`
- `slope` over time (from a linear regression on `time_step`)
- `r2` of the linear fit (stability of the trend)

These capture both the level and evolution of each counter over the vehicle’s study period.

---

### 2.3 Histogram-based features

Several sensor variables are provided as **histograms**, each represented by multiple bin columns, e.g.:

- `167_0` … `167_9`
- `272_*`, `291_*`, `158_*`, `459_*`, `397_*`

The module:

1. Groups columns into histogram families by prefix (`167_`, `272_`, etc.)
2. For each histogram family and each time-step row:
   - computes the **total mass** (sum of bins)
   - computes the **centroid** (weighted average bin index)

These per-row derived features (`prefix_total`, `prefix_centroid`) are then aggregated per vehicle.

#### Per-bin aggregations (per vehicle)

For each histogram bin column, the following are computed per vehicle:

- `mean`
- `std`
- `min`
- `max`

#### Derived histogram aggregations (per vehicle)

For each histogram family’s `total` and `centroid` columns, the module computes:

- `mean`
- `std`
- `min`
- `max`

This provides powerful descriptors of how the underlying distributions behave over time (shifts, spread, overall magnitude).

---

### 2.4 Study length features

From the TTE table, the variable:

- `length_of_study_time_step`

is brought into the feature table as:

- `study_length_time_step`

This encodes how long each vehicle has been observed (in terms of the dataset’s time_step units).

---

### 2.5 Specification encoding (categorical variables)

The specification columns:

- `Spec_0` … `Spec_7`

are categorical. The module:

- applies **one-hot encoding** using `pandas.get_dummies` with `drop_first=True`
- for the **training** split, it identifies the set of resulting one-hot columns (called `spec_feature_cols`)
- for **validation/test**, it enforces that:
  - all training-time spec columns exist (missing ones are added and filled with 0)
  - any unexpected extra categories are dropped

This guarantees that train/validation/test use the **same specification feature space**.

---

### 2.6 Final feature matrix

The final feature set per split includes:

- `vehicle_id` (identifier)
- Counter-based features (level, trend, volatility)
- Histogram bin statistics
- Histogram derived statistics (`total`, `centroid`)
- Study length
- Encoded specifications
- `in_study_repair` (target label)

For training:

- the module returns the full feature table plus:
  - `spec_feature_cols` (one-hot spec columns)
  - `feature_columns` (all feature columns excluding `vehicle_id` and the target)

For validation/test:

- the module builds features and then **aligns** them to the training `feature_columns`.

---

## 3. Building All Splits (`scripts/build_train_val_test_features.py`)

The script `scripts/build_train_val_test_features.py` orchestrates feature building for:

- **train**
- **validation**
- **test**

### 3.1 ADLS connection

The script uses `adlfs.AzureBlobFileSystem` to connect to:

- storage account: `scaniapdmstorage`
- filesystem: `scania-dataset`

Authentication is currently done using a storage **account key**:

```bash
$env:SCANIA_STORAGE_ACCOUNT_KEY = "<your-key>"
```

and read by the script via:

```python
ACCOUNT_KEY = os.environ.get("SCANIA_STORAGE_ACCOUNT_KEY", "<PASTE-KEY-HERE>")
```

### 3.2 Workflow
1. Train
* Load 3 raw train CSVs from ADLS.
* Infer histogram groups and determine which counter columns exist.
* Call build_train_features(...) from the module.
* Save train_vehicle_features.csv locally.
* Capture spec_feature_cols and feature_columns for downstream use.

2. Validation
* Load raw validation CSVs.
* Call build_eval_features(...) with:
   - the same histogram groups,
   - the same counter columns,
   - spec_feature_cols from train,
   - feature_columns from train.
* Save validation_vehicle_features.csv locally.
* Validation features are now aligned with training.

3. Test
* Same process as validation, using the test CSVs.
* Save test_vehicle_features.csv locally.

At the end of the script, engineered feature matrices for all three splits exist on the local filesystem of the environment where the script ran.

### 4. How to Run the Feature Build Script
From the repository root:
```bash
# 1. Activate environment
conda activate azure-pdm

# 2. Ensure storage key is available (temporary solution)
set SCANIA_STORAGE_ACCOUNT_KEY=<your-key>        # Windows CMD
# or
$env:SCANIA_STORAGE_ACCOUNT_KEY="<your-key>"     # PowerShell

# 3. Run the script
python scripts/build_train_val_test_features.py
```

Expected output:

* Messages about connecting to ADLS
* Shapes of raw and engineered datasets
* Three CSV files:
   - `train_vehicle_features.csv`
   - `validation_vehicle_features.csv`
   - `test_vehicle_features.csv`

### 5. Design Decisions
* No NaN imputation at this stage
  XGBoost will be used as the first baseline model, and it handles missing values natively.
  This keeps the pipeline simpler and preserves potential signal in the missingness pattern.

* Per-vehicle aggregation over sequence modeling
  Due to variable and irregular `time_step` sequences per vehicle, aggregation to a fixed-length vector per vehicle was chosen as the primary modeling approach. This is standard practice in industrial predictive maintenance   and is well-suited for tree-based models.

* Spec encoding aligned across splits
  Because specs are one-hot encoded, the script explicitly aligns validation/test features to the training feature set to avoid mismatches.

### 6. Next Steps
The next stage, described in `03_model_training.ipynb`, will:

1. Load:
* `train_vehicle_features.csv`
* `validation_vehicle_features.csv`
* `test_vehicle_features.csv`

2. Split features/labels:
* `X_train, y_train`
* `X_val, y_val`
* `X_test, y_test`

3. Train a baseline model (e.g., XGBoost) using:
* class weighting (to handle class imbalance)
* metrics such as ROC-AUC and PR-AUC.

4. Evaluate performance and document the results.

This completes the end-to-end path from raw Azure-hosted data to a trainable ML-ready dataset.
