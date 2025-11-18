# Prerequisites
Install the Azure CLI with winget: 
```powershell
winget install -e --id Microsoft.AzureCLI
Restart PowerShell and check installation:
```powershell
az --version
Add the extension:
```powershellaz extension add -n ml



# Step 1 - Authenticate with Azure

Use PowerShell to log into Azure with the following command:

```powershell
Connect-AzAccount

List available subscriptions:
```powershell
Get-AzSubscription

Select the subscription
```powershell
Set-AzContext -Subscription "Azure subscription 1"

# Step 2 - Create a Resource Group
Every Azure ML workspace must live inside a resource group.
```powershell
New-AzResourceGroup -Name "scania-pdm-rg" -Location "westeurope"

# Step 3 - Create a Storage Account (ADLS Gen2)
```powershell
New-AzStorageAccount `
  -ResourceGroupName "scania-pdm-rg" `
  -Name "scaniapdmstorage" `
  -Location "westeurope" `
  -SkuName Standard_LRS `
  -Kind StorageV2 `
  -EnableHierarchicalNamespace $true

# Step 4 - Create a Key Vault
```powershell
$keyVault = New-AzKeyVault `
    -Name "scaniapdm-kv" `
    -ResourceGroupName "scania-pdm-rg" `
    -Location "westeurope" `
    -Sku Standard

# Step 5 - Create Application Insights
```powershell
$appInsights = New-AzApplicationInsights `
    -Name "scaniapdm-ai" `
    -ResourceGroupName "scania-pdm-rg" `
    -Location "westeurope" `
    -ApplicationType web

Presupposes:
Registering the Microsoft.Insights resource provider:
```powershell
Get-AzResourceProvider -ProviderNamespace "Microsoft.Insights"

Check status with
```powershell
Get-AzResourceProvider -ProviderNamespace "Microsoft.Insights"

Similarly
```powershell
Register-AzResourceProvider -ProviderNamespace "Microsoft.MachineLearningServices"

```powershell
Register-AzResourceProvider -ProviderNamespace "Microsoft.OperationalInsights"


# Step 6 - Create the Azure ML workspace
```powershell
New-AzMLWorkspace `
    -ResourceGroupName "scania-pdm-rg" `
    -Name "scania-pdm-ws" `
    -Location "westeurope" `
    -StorageAccountId $storage.Id `
    -KeyVaultId $keyVault.ResourceId `
    -ApplicationInsightId $appInsights.Id `
    -IdentityType SystemAssigned `
    -Kind "Default"
