# azure-predictive-maintenance
A project using Azure to deploy a predictive maintenance ML project
* Overview (what problem it solves)

* Architecture diagram

* Azure services used

* Key skills demonstrated

* Setup/run instructions

* Results/screenshots

* Sources
  * Azure Machine Learning documentation: https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2
  * Datasets:
   * SCANIA Component X dataset: a real-world multivariate time series dataset for predictive maintenance: https://www.nature.com/articles/s41597-025-04802-6?fromPaywallRec=false

* Overview of repo

azure-predictive-maintenance/

├─ README.md

├─ notebooks/

│   ├─ data_preparation.ipynb

│   ├─ model_training.ipynb

├─ src/

│   ├─ train.py

│   ├─ score.py

│   ├─ utils.py

├─ pipeline/

│   ├─ azureml_pipeline.yml

├─ deployment/

│   ├─ deploy_endpoint.py

│   ├─ test_api.py

├─ data/ (or data download script)

├─ requirements.txt

└─ LICENSE
