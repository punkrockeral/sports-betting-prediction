# scripts/fetch_and_predict.py
import os
import json
import requests
from datetime import datetime, timedelta
import sys

# Fecha de hoy en formato ISO (UTC)
today = datetime.utcnow().strftime("%Y-%m-%d")
tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

# API Key (viene de GitHub Secrets como variable de entorno)
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
if not API_KEY:
    print("❌ ERROR: FOOTBALL_DATA_API_KEY no definida.")
    sys.exit(1)

headers = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api.football-data.org/v4"

# 1. Obtener partidos de hoy y mañana
fixtures = []
for date in [today, tomorrow]:
    try:
        response = requests.get(
            f"{BASE_URL}/matches",
            headers=headers,
            params={"dateFrom": date, "dateTo": date}
        )
        if response.status_code == 200:
            data = response.json()
            fixtures.extend(data.get("matches", []))
        else:
            print(f"⚠️  Error al obtener partidos para {date}: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Excepción al obtener partidos: {e}")

if not fixtures:
    print("ℹ️  No se encontraron partidos para hoy o mañana.")
    predictions = []
else:
    print(f"✅ Encontrados {len(fixtures)} partidos. Generando predicciones...")

    predictions = []
    for match in fixtures:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        match_date = match["utcDate"]

        # 2. OBTENER CUOTAS (odds) si están disponibles
        odds = {"1": 2.5, "X": 3.2, "2": 2.8}  # Valores por defecto
        try:
            odds_resp = requests.get(
                f"{BASE_URL}/matches/{match['id']}/odds",
                headers=headers
            )
            if odds_resp.status_code == 200:
                odds_data = odds_resp.json().get("odds", [])
                if odds_data and "bookmakers" in odds_data[0]:
                    # Tomar la primera casa de apuestas
                    bookie = odds_data[0]["bookmakers"][0]
                    for odd in bookie["markets"][0]["odds"]:
                        if odd["name"] == "HOME_TEAM":
                            odds["1"] = odd["odds"]
                        elif odd["name"] == "DRAW":
                            odds["X"] = odd["odds"]
                        elif odd["name"] == "AWAY_TEAM":
                            odds["2"] = odd["odds"]
        except:
            pass  # Mantener cuotas por defecto si falla

        # 3. SIMULACIÓN DE MODELO (¡aquí irá tu modelo real más adelante!)
        # Por ahora: predicción aleatoria con probabilidades simples
        prob_home = 1 / odds["1"]
        prob_draw = 1 / odds["X"]
        prob_away = 1 / odds["2"]
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total

        # Decidir resultado con mayor probabilidad
        if prob_home > prob_draw and prob_home > prob_away:
            pred = "HOME"
            ev = (prob_home * odds["1"]) - 1
        elif prob_draw > prob_away:
            pred = "DRAW"
            ev = (prob_draw * odds["X"]) - 1
        else:
            pred = "AWAY"
            ev = (prob_away * odds["2"]) - 1

        predictions.append({
            "match_id": match["id"],
            "home_team": home,
            "away_team": away,
            "match_date": match_date,
            "home_odds": odds["1"],
            "draw_odds": odds["X"],
            "away_odds": odds["2"],
            "predicted_outcome": pred,
            "expected_value": round(ev, 3)
        })

# 4. Guardar resultados
os.makedirs("data/predictions", exist_ok=True)
output_path = f"data/predictions/{today}.json"
with open(output_path, "w") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "date": today,
        "predictions": predictions
    }, f, indent=2)

print(f"✅ Predicciones guardadas en {output_path}")