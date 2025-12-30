# APIs para Football-Data.org
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
