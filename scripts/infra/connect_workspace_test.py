from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

SUBSCRIPTION_ID = "<YOUR-SUBSCRIPTION-ID>"
RESOURCE_GROUP = "scania-pdm-rg"
WORKSPACE_NAME = "scania-pdm-ws"

def main():
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    # Simple sanity check: print workspace details
    ws = ml_client.workspaces.get(name=WORKSPACE_NAME)
    print(f"Connected to workspace: {ws.name} in {ws.location}")

if __name__ == "__main__":
    main()
