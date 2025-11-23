import pandas as pd
import json
import requests

# Fill with values from Azure ML Studio
SCORING_URL = "https://scania-pdm-20251123130737.westeurope.inference.ml.azure.com/score" #"<endpoint-scoring-url>"
API_KEY = "LASe471Gea5ZvamYR4EZkVON32iS2AoDYqh6NO7p1vXI9dLKwl50JQQJ99BKAAAAAAAAAAAAINFRAZML11FR" #"<primary-key>"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# Example payload
# 1. Load features CSV
# Adjust path if needed
df = pd.read_csv("/home/azureuser/cloudfiles/code/users/rammekjaer/train_vehicle_features.csv")

# 2. Drop non-feature columns
drop_cols = ["vehicle_id", "in_study_repair"]
feature_cols = [c for c in df.columns if c not in drop_cols]

# 3. Pick a random row
sample_row = df.sample(n=1, random_state=42)[feature_cols].iloc[0]

# 4. Convert to the payload format expected by score.py
payload = {
    "data": [
        sample_row.to_dict()
    ]
}
response = requests.post(SCORING_URL, headers=headers, data=json.dumps(payload))

print("Status code:", response.status_code)
print("Response:", response.json())
