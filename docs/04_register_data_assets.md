# Register Data Assets in Azure ML

This document describes how to register the SCANIA predictive maintenance data as Azure ML Data Assets, using the files previously uploaded to the Azure Data Lake Storage account scaniapdmstorage.

Azure ML Data Assets allow training jobs and pipelines to reference versioned datasets without manually handling file paths.

## Overview
We register three Data Assets:
| Name                    | Type       | Source path                  |
| ----------------------- | ---------- | ---------------------------- |
| `scania_raw_train`      | URI folder | `scania-dataset/train/`      |
| `scania_raw_validation` | URI folder | `scania-dataset/validation/` |
| `scania_raw_test`       | URI folder | `scania-dataset/test/`       |

All of these point into the Azure ML datastore we registered earlier:
```bash
azureml://datastores/scaniadatalake/paths/<path>
```
## Prerequisites
Before running the script:

* The ML workspace exists (scania-pdm-ws)
* The datastore has been registered (scaniadatalake)
* The data has been uploaded into ADLS Gen2 under the filesystem scania-dataset
* The local azure-pdm conda environment is active:
```bash
conda activate azure-pdm
  ```

## Step 1 - Inspect folder structure in Storage
```bash
scaniapdmstorage
└── scania-dataset/
    ├── train/
    ├── validation/
    └── test/
```
## Step 2 - Run the registration script
```powershell
python ./scripts/infra/register_data_assets.py
```

## Step 3 - Verify in Azure ML Studio

Go to:

Azure ML Studio → Data → Data assets

You should now see:

* scania_raw_train (version 1)
* scania_raw_validation (version 1)
* scania_raw_test (version 1)

Click one to confirm:

* Path: azureml://datastores/scaniadatalake/...
* Type: uri_folder
* Version: 1
* File preview works

## Completion

The SCANIA dataset is now:

* stored in ADLS Gen2
* organized into train/validation/test splits
* registered as versioned Data Assets
* ready to be used in notebooks, training jobs, and pipelines

## Check
You can open a notebook in the scania-pdm-ws workspace and check the availability of the data.
E.g.
```python
%pip install azure-ai-ml azure-identity adlfs
```

```python
SUBSCRIPTION_ID = "<SUBSCRIPTION-ID-HERE>"
RESOURCE_GROUP = "scania-pdm-rg"
WORKSPACE_NAME = "scania-pdm-ws"

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

data_asset = ml_client.data.get(name="scania_raw_train", version="1")
print(data_asset.path)
```

```python
from adlfs import AzureBlobFileSystem
import pandas as pd

fs = AzureBlobFileSystem(
    account_name="scaniapdmstorage",
    account_key="<ACCOUNT-KEY-HERE>",
)

csv_path = "scania-dataset/train/train_operational_readouts.csv"

with fs.open(csv_path, "rb") as f:
    df = pd.read_csv(f)

df.head()
```
