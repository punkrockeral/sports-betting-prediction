import json
import os
from datetime import datetime

today = datetime.utcnow().strftime("%Y-%m-%d")
matches = [
    {"home": "Team A", "away": "Team B", "pred": "HOME", "ev": 0.12},
    {"home": "Team C", "away": "Team D", "pred": "DRAW", "ev": 0.08},
]

os.makedirs("data/predictions", exist_ok=True)
output_path = f"data/predictions/{today}.json"

with open(output_path, "w") as f:
    json.dump({
        "date": today,
        "predictions": matches
    }, f, indent=2)

print(f"✅ Predicciones guardadas en {output_path}")