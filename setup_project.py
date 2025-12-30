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

# Definir archivos a crear con contenido inicial (opcional)
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
    url = f"https://api.football-data.org/v4/matches?dateFrom={date}&dateTo={date}"
    resp = requests.get(url, headers=headers)
    return resp.json() if resp.status_code == 200 else {}
""",

    "apis/odds_api.py": """# APIs para The Odds API
import requests
import os

def get_odds(sport="soccer_epl", regions="eu", markets="h2h"):
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY no configurada")
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal"
    }
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
    st.subheader(f"Predicciones del {data['date']}")
    for p in data.get("predictions", []):
        ev = p.get("expected_value", 0)
        if ev > 0.1:
            st.write(f"**{p['home_team']} vs {p['away_team']}** → {p['predicted_outcome']} (EV: {ev:.2%})")
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
from apis.odds_api import