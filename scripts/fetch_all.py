# scripts/fetch_all.py
import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Configuración
TODAY = datetime.utcnow().strftime("%Y-%m-%d")
SPORTS_ODDS = ["soccer_epl", "soccer_spain_la_liga", "soccer_germany_bundesliga"]
LEAGUES_FD = ["PL", "PD", "BL"]  # Códigos de Football-Data

def fetch_football_data():
    """Obtiene fixtures y odds de Football-Data.org"""
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print("⚠️  FOOTBALL_DATA_API_KEY no configurada.")
        return {}

    all_data = {}
    for league_code in LEAGUES_FD:
        try:
            # Fixtures
            fixtures_url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
            headers = {"X-Auth-Token": api_key}
            resp = requests.get(fixtures_url, headers=headers, params={"dateFrom": TODAY, "dateTo": TODAY})
            if resp.status_code == 200:
                matches = resp.json().get("matches", [])
                # Obtener odds para cada match
                for match in matches:
                    try:
                        odds_url = f"https://api.football-data.org/v4/matches/{match['id']}/odds"
                        odds_resp = requests.get(odds_url, headers=headers)
                        if odds_resp.status_code == 200:
                            match["odds_data"] = odds_resp.json()
                    except Exception as e:
                        match["odds_data"] = {"error": str(e)}
                all_data[league_code] = matches
        except Exception as e:
            print(f"❌ Error en Football-Data ({league_code}): {e}")
    return all_data

def fetch_odds_api():
    """Obtiene cuotas en tiempo real de múltiples casas (The Odds API)"""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("⚠️  ODDS_API_KEY no configurada.")
        return {}

    all_odds = {}
    for sport in SPORTS_ODDS:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            params = {
                "apiKey": api_key,
                "regions": "eu",          # Europa
                "markets": "h2h",         # Moneyline
                "oddsFormat": "decimal",
                "bookmakers": "pinnacle,bet365"  # Casas clave
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                all_odds[sport] = resp.json()
            else:
                print(f"⚠️  The Odds API error ({sport}): {resp.status_code}")
        except Exception as e:
            print(f"❌ Error en The Odds API ({sport}): {e}")
    return all_odds

def main():
    print(f"📡 Recopilando datos deportivos para {TODAY}...")
    
    data = {
        "football_data": fetch_football_data(),
        "odds_api": fetch_odds_api(),
        "collected_at": datetime.utcnow().isoformat() + "Z"
    }

    # Guardar
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{TODAY}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Datos guardados en {output_path}")
    print(f"📊 Partidos recopilados: {len(data)} fuentes")

if __name__ == "__main__":
    main()