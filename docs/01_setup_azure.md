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
New-AzResourceGroup `
    -Name "scania-pdm-rg" `
    -Location "westeurope"

# Step 3 - Create the Azure ML Workspace
```powershell
New-AzMlWorkspace `
    -Name "scania-pdm-ws" `
    -ResourceGroupName "scania-pdm-rg" `
    -Location "westeurope"

This requires
```powershell
Install-Module -Name Az.MachineLearningServices

```powershell
