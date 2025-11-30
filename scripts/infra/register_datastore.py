from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureDataLakeGen2Datastore

SUBSCRIPTION_ID = "<YOUR-SUBSCRIPTION-ID>"
RESOURCE_GROUP = "scania-pdm-rg"
WORKSPACE_NAME = "scania-pdm-ws"

STORAGE_ACCOUNT_NAME = "scaniapdmstorage"
FILESYSTEM_NAME = "scania-dataset"  # container/filesystem created for the data

def main():
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    datastore = AzureDataLakeGen2Datastore(
        name="scaniadatalake",
        description="ADLS Gen2 datastore for SCANIA PdM dataset",
        account_name=STORAGE_ACCOUNT_NAME,
        filesystem=FILESYSTEM_NAME,
    )

    ml_client.datastores.create_or_update(datastore)
    print("✅ Datastore 'scaniadatalake' registered successfully.")

if __name__ == "__main__":
    main()
