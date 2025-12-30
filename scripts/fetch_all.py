# scripts/fetch_all.py
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
