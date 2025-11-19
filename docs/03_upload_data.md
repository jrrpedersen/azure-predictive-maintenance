
# Upload data
## Step 1 - Create ML handle
See script connect_workspace_test.py

## Step 2 - Get storage account key
Go to your storage account scaniapdmstorage

Left menu → Security + networking → Access keys

Copy one of the keys (Key1 is fine)

Set as an environmental variable
```powershell
$env:SCANIA_STORAGE_ACCOUNT_KEY = "<paste-key-here>"
```