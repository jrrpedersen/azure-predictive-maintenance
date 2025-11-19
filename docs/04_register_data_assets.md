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
### Step 3 - Run the registration script
```powershell
python ./scripts/infra/register_data_assets.py
```

## Step Step 4 — Verify in Azure ML Studio

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
