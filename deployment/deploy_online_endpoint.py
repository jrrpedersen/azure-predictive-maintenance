# Run this on compute instance to deploy the online endpoint:
# cd cloudfiles/code/azure-predictive-maintenance/deployment
# python deploy_online_endpoint.py

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    Model,
    CodeConfiguration,
)

import datetime

# Workspace config - fill in your values
SUBSCRIPTION_ID = "52124f51-27f5-4c3c-99a9-fa716e4e8cfe"
RESOURCE_GROUP = "scania-pdm-rg"
WORKSPACE_NAME = "scania-pdm-ws"

# Name of the registered tuned model
REGISTERED_MODEL_NAME = "scania-pdm-xgb-finetuned"  # adjust if needed

# Endpoint and deployment names (must be unique in workspace)
timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
ENDPOINT_NAME = f"scania-pdm-{timestamp.lower()}"
DEPLOYMENT_NAME = "xgb-tuned-deployment"

# 1. Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

print("Connected to workspace:", WORKSPACE_NAME)

# 2. Reference the registered model
model = ml_client.models.get(name=REGISTERED_MODEL_NAME, version=1)
print("Using registered model:", model.name, "version:", model.version)

# 3. Define environment for deployment
env = Environment(
    name="scania-pdm-endpoint-env",
    description="Environment for Scania PDM XGBoost endpoint",
    conda_file="conda.yaml",
    #image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    image="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest"
)

# 4. Create endpoint definition
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    auth_mode="key",  # or "aml_token"
    description="Online endpoint for Scania PdM tuned XGBoost model",
)

print("Creating endpoint:", ENDPOINT_NAME)
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint created.")

# 5. Create deployment
deployment = ManagedOnlineDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code=".",            # directory containing score.py and model file
        scoring_script="score.py",
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

print("Creating deployment:", DEPLOYMENT_NAME)
deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
print("Deployment created.")

# 6. Route 100% of traffic to this deployment
endpoint.traffic = {DEPLOYMENT_NAME: 100}
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Traffic updated. 100% ->", DEPLOYMENT_NAME)

print("\n✅ Endpoint is ready.")
print("Endpoint name:", ENDPOINT_NAME)
