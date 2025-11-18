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

Using VS Code with the Github Pull extension
- Download and install Git from: https://git-scm.com/download/win
- Download VS Code: https://code.visualstudio.com/
- Install the GitHub extension
- Clone GitHub repo using VS Code
- Open a terminal inside VS Code: conda activate azure-pdm
