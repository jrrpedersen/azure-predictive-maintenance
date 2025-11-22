# deployment/score.py

import json
import joblib
import numpy as np
import pandas as pd

model = None
BEST_THRESHOLD = 0.51
FEATURE_COLS = None

def init():
    # global model, FEATURE_COLS
    # model_path = "xgb_pdm_finetuned.pkl"
    # model = joblib.load(model_path)

    # # Load feature column order
    # with open("feature_cols.json", "r") as f:
    #     FEATURE_COLS = json.load(f)

    # print("Model loaded from", model_path)
    # print("Loaded", len(FEATURE_COLS), "feature columns.")
    # print("Using BEST_THRESHOLD =", BEST_THRESHOLD)
    global model, FEATURE_COLS

    # Where Azure ML mounts the registered model
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "xgb_pdm_finetuned.pkl")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # feature_cols.json should live next to score.py in your deployment folder
    feature_cols_path = os.path.join(os.path.dirname(__file__), "feature_cols.json")
    print(f"Loading feature columns from: {feature_cols_path}")

    with open(feature_cols_path, "r") as f:
        FEATURE_COLS = json.load(f)

    print(f"Loaded {len(FEATURE_COLS)} feature columns.")
    print(f"BEST_THRESHOLD = {BEST_THRESHOLD}")

def run(raw_data):
    try:
        data = json.loads(raw_data)

        if "data" not in data:
            return json.dumps({"error": "Request JSON must contain a 'data' field."})

        df = pd.DataFrame(data["data"])

        # Enforce same column set and order as training
        missing = set(FEATURE_COLS) - set(df.columns)
        extra = set(df.columns) - set(FEATURE_COLS)

        if missing:
            return json.dumps({"error": f"Missing features: {sorted(missing)}"})
        if extra:
            # You can ignore or warn about extra columns
            df = df[FEATURE_COLS]
        else:
            df = df[FEATURE_COLS]

        proba = model.predict_proba(df)[:, 1]
        labels = (proba >= BEST_THRESHOLD).astype(int)

        results = [
            {
                "failure_probability": float(p),
                "failure_imminent": bool(l),
                "threshold_used": BEST_THRESHOLD,
            }
            for p, l in zip(proba, labels)
        ]

        return json.dumps({"results": results})

    except Exception as e:
        error_message = f"Error during scoring: {str(e)}"
        print(error_message)
        return json.dumps({"error": error_message})
