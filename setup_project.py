# setup_project.py
import os
from pathlib import Path

# Definir la estructura de carpetas
dirs = [
    "data/raw",
    "data/processed",
    "data/predictions",
    "models",
    "apis",
    "core",
    "app",
    "scripts",
    ".github/workflows"
]

# Definir archivos con contenido seguro (sin errores de sintaxis)
files = {
    "requirements.txt": """# Core
pandas
numpy
scikit-learn
xgboost
requests
python-dateutil

# Modelos
tensorflow
joblib

# Dashboard
streamlit

# APIs
python-dotenv
""",

    "config.yaml": """# Configuración del sistema
sports:
  - soccer_epl
  - soccer_spain_la_liga
  - basketball_nba

thresholds:
  min_ev: 0.1
  max_odds: 10.0

apis:
  football_data_enabled: true
  odds_api_enabled: true
""",

    "apis/football_data.py": """# APIs para Football-Data.org
import requests
import os

def get_fixtures(date):
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise ValueError("FOOTBALL_DATA_API_KEY no configurada")
    headers = {"X-Auth-Token": api_key}
    url = f"https://api.football-data.org/v4/matches?dateFrom={{date}}&dateTo={{date}}"
    resp = requests.get(url, headers=headers)
    return resp.json() if resp.status_code == 200 else {{}}
""",

    "apis/odds_api.py": """# APIs para The Odds API
import requests
import os

def get_odds(sport="soccer_epl", regions="eu", markets="h2h"):
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY no configurada")
    url = f"https://api.the-odds-api.com/v4/sports/{{sport}}/odds"
    params = {{
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal"
    }}
    resp = requests.get(url, params=params)
    return resp.json() if resp.status_code == 200 else []
""",

    "app/dashboard.py": """import streamlit as st
import json
import glob

st.set_page_config(page_title="Betting Predictor", layout="wide")
st.title("🎯 Predicción de Apuestas Deportivas")

pred_files = sorted(glob.glob("data/predictions/*.json"), reverse=True)
if pred_files:
    with open(pred_files[0]) as f:
        data = json.load(f)
    st.subheader(f"Predicciones del {{data['date']}}")
    for p in data.get("predictions", []):
        ev = p.get("expected_value", 0)
        if ev > 0.1:
            st.write(f"**{{p['home_team']}} vs {{p['away_team']}}** → {{p['predicted_outcome']}} (EV: {{ev:.2%}})")
else:
    st.warning("No hay predicciones aún.")
""",

    "core/predictor.py": """# Lógica central de predicción
def predict_match(home_features, away_features, model):
    # Placeholder: aquí irá tu modelo real
    return {"outcome": "HOME", "prob": 0.45}
""",

    "scripts/fetch_all.py": """# scripts/fetch_all.py
import os
import json
from datetime import datetime
from apis.football_data import get_fixtures
from apis.odds_api import get_odds

def main():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    data = {}
    
    # Football-Data
    if os.getenv("FOOTBALL_DATA_API_KEY"):
        data["football_data"] = get_fixtures(today)
    
    # The Odds API
    if os.getenv("ODDS_API_KEY"):
        data["odds_api"] = get_odds(sport="soccer_epl")
    
    # Guardar
    os.makedirs("data/raw", exist_ok=True)
    with open(f"data/raw/{{today}}.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Datos guardados para {{today}}")

if __name__ == "__main__":
    main()
""",

    ".github/workflows/daily_predict.yml": """name: Daily Prediction

on:
  schedule:
    - cron: '0 6 * * *'
  workflow_dispatch:

jobs:
  predict:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt

      - name: Run prediction
        env:
          FOOTBALL_DATA_API_KEY: ${{ secrets.FOOTBALL_DATA_API_KEY }}
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
        run: python scripts/fetch_all.py && python scripts/generate_daily_preds.py

      - name: Commit predictions
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add data/predictions/
          if ! git diff --staged --quiet; then
            git commit -m "Auto: Predicciones del $(date -u +%Y-%m-%d)"
            git remote set-url origin https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/punkrockeral/sports-betting-prediction.git
            git push
          fi
"""
}

# Crear carpetas
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"📁 Carpeta creada: {d}")

# Crear archivos
for filepath, content in files.items():
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Asegurar que el contenido termine con salto de línea
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        print(f"📄 Archivo creado: {filepath}")
    else:
        print(f"⏭️  Archivo ya existe: {filepath}")

print("\n✅ ¡Estructura del proyecto generada con éxito!")
print("👉 Siguiente paso: añade tus claves API en GitHub Secrets y personaliza los scripts.")