# APIs para The Odds API
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
