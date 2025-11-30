from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# ----------------------------
# Workspace configuration
# ----------------------------
SUBSCRIPTION_ID = "<YOUR-SUBSCRIPTION-ID>"   # GUID
RESOURCE_GROUP = "scania-pdm-rg"
WORKSPACE_NAME = "scania-pdm-ws"

# ----------------------------
# Datastore + folder paths
# ----------------------------
DATASTORE_NAME = "scaniadatalake"
BASE_PATH = "scania-dataset"  # ADLS filesystem root folder

DATASETS = {
    "scania_raw_train": "train",
    "scania_raw_validation": "validation",
    "scania_raw_test": "test",
}

def main():
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    for name, subfolder in DATASETS.items():
        path = f"azureml://datastores/{DATASTORE_NAME}/paths/{BASE_PATH}/{subfolder}/"

        data_asset = Data(
            name=name,
            version="1",
            type=AssetTypes.URI_FOLDER,
            path=path,
            description=f"SCANIA PdM dataset: {subfolder} split",
        )

        created = ml_client.data.create_or_update(data_asset)
        print(f"✅ Registered data asset: {created.name}, version {created.version}")
        print(f"   → {created.path}")

if __name__ == "__main__":
    main()
