# Register scaniapdmstorage (data lake) as a datastore inside the workspace

Create env
```bash
conda create -n azure-pdm python=3.10
```
```bash
conda activate azure-pdm
```


Install the Azure ML SDK (v2) and dependencies:
```bash
pip install azure-ai-ml azure-identity
```
Install typical ML/data stack:
```bash
pip install pandas numpy scikit-learn matplotlib
```
