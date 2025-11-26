
# 10 – Online Inference & Endpoint Usage

This document explains how to use the **Scania PdM online endpoint** to obtain predictions from the tuned XGBoost model, both from **inside Azure ML** (SDK) and from **external clients** (raw HTTP via `requests`).

The endpoint serves the model `scania-pdm-xgb-finetuned` and returns:

- A **failure probability** per vehicle
- A binary decision `failure_imminent` based on a **cost-optimized threshold** `τ = 0.51`

---

## 1. Endpoint overview

The endpoint is a **Managed Online Endpoint** deployed in the workspace:

- **Subscription:** `52124f51-27f5-4c3c-99a9-fa716e4e8cfe`
- **Resource group:** `scania-pdm-rg`
- **Workspace:** `scania-pdm-ws`
- **Endpoint name pattern:** `scania-pdm-<timestamp>`
- **Deployment name:** `xgb-tuned-deployment`
- **Auth mode:** `key`

The deployment uses:

- Model: `scania-pdm-xgb-finetuned` (registered in Azure ML)
- Scoring script: `deployment/score.py`
- Environment: `deployment/conda.yaml`
- Inference image: `mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest`

The scoring script:

- Loads the tuned XGBoost model (`xgb_pdm_finetuned.pkl`)
- Expects a JSON payload with a `data` list of feature dictionaries
- Applies the threshold `0.51` to convert probabilities into binary labels

---

## 2. Request & response schema

### 2.1 Request format

The endpoint expects a JSON object with a single key: **`"data"`**, containing a list of rows.  
Each row is a dictionary mapping **feature name → value**.

Example:

```json
{
  "data": [
    {
      "feature_1": 0.123,
      "feature_2": 4.56,
      "...": 0.0
    }
  ]
}
```

In practice, the feature names must match the **engineered feature columns** from `train_vehicle_features.csv`, excluding:

- `vehicle_id`
- `in_study_repair`

For testing, you can construct a payload by:

1. Loading `train_vehicle_features.csv`
2. Dropping `vehicle_id` and `in_study_repair`
3. Sampling a row and converting it to a dict

Example (see also `deployment/test_endpoint.py`):

```python
import pandas as pd

df = pd.read_csv("train_vehicle_features.csv")
drop_cols = ["vehicle_id", "in_study_repair"]
feature_cols = [c for c in df.columns if c not in drop_cols]

sample_row = df.sample(n=1, random_state=42)[feature_cols].iloc[0]
payload = {"data": [sample_row.to_dict()]}
```

### 2.2 Response format

The endpoint returns a JSON object with a **`results`** list, one item per input row:

```json
{
  "results": [
    {
      "failure_probability": 0.0419,
      "failure_imminent": false,
      "threshold_used": 0.51
    }
  ]
}
```

- `failure_probability`: model’s predicted probability of imminent failure
- `failure_imminent`: `true` if `probability >= 0.51`, else `false`
- `threshold_used`: the threshold used to make the decision (currently `0.51`)

---

## 3. Getting the scoring URL & API key

You need two pieces of information for external calls:

- **SCORING_URL** – the REST endpoint URI
- **API_KEY** – primary or secondary key

### 3.1 Via Azure ML Studio (UI)

1. Open **Azure ML Studio** and go to **Endpoints**.
2. Click the **online endpoint** you deployed (e.g. `scania-pdm-20251123130737`).
3. Go to the **Consume** (or **Details**) tab.
4. Copy:
   - **REST endpoint** → `SCORING_URL`
   - **Primary key** → `API_KEY`

### 3.2 Via Python SDK

From an authenticated environment (e.g. your compute instance):

```python
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

SUBSCRIPTION_ID = "52124f51-27f5-4c3c-99a9-fa716e4e8cfe"
RESOURCE_GROUP = "scania-pdm-rg"
WORKSPACE_NAME = "scania-pdm-ws"
ENDPOINT_NAME = "<your-endpoint-name>"

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
print("Scoring URI:", endpoint.scoring_uri)

keys = ml_client.online_endpoints.list_keys(name=ENDPOINT_NAME)
print("Primary key:", keys.primary_key)
print("Secondary key:", keys.secondary_key)
```

---

## 4. Invoking the endpoint from Azure (SDK)

For quick tests from your compute instance or any authenticated Python environment, you can use the Azure ML SDK’s `invoke` method.

First, create a `sample-request.json` file, e.g.:

```json
{
  "data": [
    {
      "feat_1": 0.1,
      "feat_2": 3.14
      // ...
    }
  ]
}
```

Then call:

```python
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

SUBSCRIPTION_ID = "52124f51-27f5-4c3c-99a9-fa716e4e8cfe"
RESOURCE_GROUP = "scania-pdm-rg"
WORKSPACE_NAME = "scania-pdm-ws"
ENDPOINT_NAME = "<your-endpoint-name>"

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

response = ml_client.online_endpoints.invoke(
    endpoint_name=ENDPOINT_NAME,
    deployment_name="xgb-tuned-deployment",
    request_file="sample-request.json",
)

print(response)
```

This is useful when working fully inside Azure.

---

## 5. Invoking the endpoint from an external client (Python `requests`)

This is the pattern a real application (e.g. microservice or batch job) would use.

Example: `deployment/test_endpoint.py`:

```python
import pandas as pd
import json
import requests

SCORING_URL = "https://<your-endpoint-name>.<region>.inference.ml.azure.com/score"
API_KEY = "<your-primary-key>"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# Load engineered features
df = pd.read_csv("/path/to/train_vehicle_features.csv")

# Drop label/id columns
drop_cols = ["vehicle_id", "in_study_repair"]
feature_cols = [c for c in df.columns if c not in drop_cols]

# Pick a random example row
sample_row = df.sample(n=1, random_state=42)[feature_cols].iloc[0]

# Build payload
payload = {
    "data": [
        sample_row.to_dict()
    ]
}

response = requests.post(SCORING_URL, headers=headers, data=json.dumps(payload))

print("Status code:", response.status_code)
print("Response:", response.json())
```

Typical output:

```text
Status code: 200
Response: {
  "results": [
    {
      "failure_probability": 0.0419,
      "failure_imminent": false,
      "threshold_used": 0.51
    }
  ]
}
```

---

## 6. How this fits into the overall project

By deploying the tuned XGBoost model as an online endpoint and successfully invoking it, the project now demonstrates an **end-to-end MLOps flow**:

1. Data ingestion and storage in ADLS Gen2  
2. Feature engineering into per-vehicle feature matrices  
3. Model training, tuning, and cost-based threshold selection  
4. Model registration in Azure ML  
5. Online endpoint deployment with `score.py` and a custom environment  
6. Real-time scoring via REST API

This closes the loop from raw data to a callable AI service.
