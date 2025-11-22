To get SCORING_URL and API_KEY:
**A Azure ML Studio (UI)**:
1. Go to Azure ML Studio → Endpoints.
2. Click your online endpoint (the one you created with the deployment script).
3. In the left panel (or top tabs), click “Consume” or “Details” (depends on UI version).
  * REST endpoint → this is your SCORING_URL
  * Primary key / Secondary key → one of these is your API_KEY

**B Via the Python SDK**
From the compute instance, run:
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

```python
import json
import requests

# Fill with values from Azure ML Studio
SCORING_URL = "<endpoint-scoring-url>"
API_KEY = "<primary-key>"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# Example payload: one pre-computed feature row (shortened)
payload = {
    "data": [
        {
            # Example fields – must match your feature names:
            # "feature1": 0.123,
            # "feature2": 4.56,
            # ...
        }
    ]
}

response = requests.post(SCORING_URL, headers=headers, data=json.dumps(payload))

print("Status code:", response.status_code)
print("Response:", response.json())
```
The response should look like:
```python
{
  "results": [
    {
      "failure_probability": 0.237,
      "failure_imminent": false,
      "threshold_used": 0.51
    }
  ]
}

```
