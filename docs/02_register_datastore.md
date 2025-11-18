# Register `scaniapdmstorage` (data lake) as a datastore in the Azure ML workspace

This page describes how to register the ADLS Gen2 storage account `scaniapdmstorage` as a datastore in the `scania-pdm-ws` Azure ML workspace using a local Python script.

## Prerequisites

Before you run these steps, you should have:

- An Azure subscription (with Contributor access).
- A resource group: `scania-pdm-rg`.
- An Azure ML workspace: `scania-pdm-ws`.
- An ADLS Gen2 storage account: `scaniapdmstorage`.
- This repo cloned locally (e.g. to `C:\Users\Administrator\github\azure-pdm`).
- The script `scripts/infra/register_datastore.py` present in the repo.

In `register_datastore.py`, make sure you set:

- `SUBSCRIPTION_ID` – the actual subscription **GUID** from the Azure Portal.
- `RESOURCE_GROUP` – e.g. `"scania-pdm-rg"`.
- `WORKSPACE_NAME` – e.g. `"scania-pdm-ws"`.
- `STORAGE_ACCOUNT_NAME` – e.g. `"scaniapdmstorage"`.
- `FILESYSTEM_NAME` – the container/filesystem name you’ll use for the dataset, e.g. `"scania-dataset"`.

---

## Step 1 — Create and activate a local conda environment

Create the environment:

```bash
conda create -n azure-pdm python=3.10
```
Activate it:
```bash
conda activate azure-pdm
```
Install the Azure ML SDK (v2) and authentication dependency:
```bash
pip install azure-ai-ml azure-identity
```
Install typical ML/data stack:
```bash
pip install pandas numpy scikit-learn matplotlib
```
Export the environment for reproducibility:
```bash
conda env export > environment.yml
```

## Step 2 - Open the project in VS Code
1. Install Git: https://git-scm.com/download/win

2. Install VS Code: https://code.visualstudio.com/

3. In VS Code, install the GitHub Pull Requests and Issues extension (optional but nice).

4. Clone this GitHub repo using VS Code (Git: Clone).

5. Open a terminal inside VS Code (View → Terminal) and activate the env
```bash
conda activate azure-pdm
```
Make sure your current directory is the repo root, e.g.:
```bash
cd C:\Users\Administrator\github\azure-pdm
```

## Step 3 - Register ADLS Gen2 datastore in Azure ML
The script uses DefaultAzureCredential, which will typically pick up an interactive browser login or an existing Azure CLI login.

If needed, log in via the browser when prompted.

Run the datastore registration script:
```bash
python scripts/infra/register_datastore.py
```