import os
from pathlib import Path

from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import ResourceExistsError

# ---- Config ----
ACCOUNT_NAME = "scaniapdmstorage"          # your ADLS Gen2 account
FILE_SYSTEM_NAME = "scania-dataset"        # filesystem / container name to create/use

# Local base path where the CSVs live (relative to repo root)
LOCAL_BASE = Path("data") / "scania"

# Expected files per split
FILES = {
    "train": [
        "train_operational_readouts.csv",
        "train_tte.csv",
        "train_specifications.csv",
    ],
    "validation": [
        "validation_operational_readouts.csv",
        "validation_labels.csv",
        "validation_specifications.csv",
    ],
    "test": [
        "test_operational_readouts.csv",
        "test_labels.csv",
        "test_specifications.csv",
    ],
}


def get_service_client() -> DataLakeServiceClient:
    """Create a DataLakeServiceClient using account key from env var."""
    account_key = os.environ.get("SCANIA_STORAGE_ACCOUNT_KEY")
    if not account_key:
        raise RuntimeError(
            "Environment variable SCANIA_STORAGE_ACCOUNT_KEY is not set. "
            "Set it in your shell before running this script."
        )

    account_url = f"https://{ACCOUNT_NAME}.dfs.core.windows.net"
    return DataLakeServiceClient(account_url=account_url, credential=account_key)


def ensure_filesystem(service_client: DataLakeServiceClient):
    """Create the filesystem if it does not exist."""
    fs_client = service_client.get_file_system_client(FILE_SYSTEM_NAME)
    try:
        fs_client.create_file_system()
        print(f"Created filesystem: {FILE_SYSTEM_NAME}")
    except ResourceExistsError:
        print(f"Filesystem already exists: {FILE_SYSTEM_NAME}")
    return fs_client


def upload_split(fs_client, split: str, filenames: list[str]):
    """Upload all CSVs for a given split (train/validation/test)."""
    local_dir = LOCAL_BASE / split
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    for name in filenames:
        local_path = local_dir / name
        if not local_path.exists():
            raise FileNotFoundError(f"Expected file not found: {local_path}")

        remote_path = f"{split}/{name}"  # e.g. train/train_operational_readouts.csv
        print(f"Uploading {local_path} -> {remote_path}")

        file_client = fs_client.get_file_client(remote_path)
        with open(local_path, "rb") as f:
            file_client.upload_data(f, overwrite=True)

    print(f"✅ Uploaded {split} files.")


def main():
    service_client = get_service_client()
    fs_client = ensure_filesystem(service_client)

    for split, filenames in FILES.items():
        upload_split(fs_client, split, filenames)

    print("✅ All splits uploaded successfully.")


if __name__ == "__main__":
    main()
