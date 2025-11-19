# Upload Data to Azure Data Lake (scaniapdmstorage)
This document describes how to upload the SCANIA predictive maintenance dataset from your local machine into the Azure Data Lake Storage Gen2 account scaniapdmstorage.
The dataset is never stored in GitHub due to size limits and best practices for ML projects. Thus, the following is added to .gitignore:
```bash
# Ignore local data files to avoid committing large datasets
azure-predictive-maintenance/data/scania/**/*.csv
```
## Overview
```bash
scaniapdmstorage
└── scania-dataset/           (filesystem / container)
    ├── train/
    ├── validation/
    └── test/
```
Each folder contains 3 CSV files:

* operational readouts
* time-to-event (TTE)
* specifications

After upload, these data folders will be used to register Azure ML Data Assets.

## Step 1 - Create ML workspace handle (optional check)
Verify that the local Python environment can connect to the ML workspace.
Run
```powershell
python ./scripts/infra/connect_workspace_test.py
```
Expected output:
Run
```powershell
Connected to workspace: scania-pdm-ws in westeurope
```
If this works, your local environment and Azure credentials are configured correctly.

## Step 2 - Get and set the Storage Account Key
The upload script uses a storage account key to authenticate.
1. Go to your storage account scaniapdmstorage
2. Left menu → Security + networking → Access keys
3. Copy one of the keys (Key1 is fine)

Set it as an environment variable in your PowerShell terminal:
```powershell
$env:SCANIA_STORAGE_ACCOUNT_KEY = "<paste-key-here>"
```
The upload script reads it from the environment at runtime.

## Step 3 - Install required Python packages
Ensure the Azure Data Lake SDK is installed in your active conda environment (azure-pdm):
```powershell
pip install azure-storage-file-datalake
```

## Step 4 - Prepare the local dataset folder
Place the CSV files locally in the following structure:
```bash
azure-predictive-maintenance/
└── data/
    └── scania/
        ├── train/
        │   ├── train_operational_readouts.csv
        │   ├── train_tte.csv
        │   └── train_specifications.csv
        ├── validation/
        │   ├── validation_operational_readouts.csv
        │   ├── validation_tte.csv
        │   └── validation_specifications.csv
        └── test/
            ├── test_operational_readouts.csv
            ├── test_tte.csv
            └── test_specifications.csv

```
These local files are ignored by Git using .gitignore, and are only used for upload.

## Step 5 - Upload data
The script uploads the contents of:
```bash
data/scania/train/
data/scania/validation/
data/scania/test/
```
into the Azure Data Lake filesystem:
```bash
scania-dataset/
```
Make sure the datastore is registered (previous step). Then run the script upload_scania_data.py:
```powershell
./scripts/infra/upload_scania_data.py
```
When complete, the data is available in Azure and ready for Data Asset registration.

## Step 6 - Verify upload in Azure Portal
1. Go to scaniapdmstorage
2. Open Containers / File systems
3. Open the filesystem scania-dataset
4. Confirm the folder structure:
```bash
train/ (3 files)
validation/ (3 files)
test/ (3 files)
```
