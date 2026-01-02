# scripts/download_historical_data.py
import pandas as pd
import os
import requests
from datetime import datetime

def download_season(league, season, output_path):
    """Descarga una temporada desde football-data.co.uk"""
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    try:
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Descargado: {output_path} ({len(df)} partidos)")
        return True
    except Exception as e:
        print(f"⚠️  Error al descargar {url}: {e}")
        return False

def main():
    seasons = {
        "2324": "E0",  # Premier League 2023/24
        "2223": "E0",  # Premier League 2022/23
        "2122": "E0",  # Premier League 2021/22
        "2021": "E0",  # Premier League 2020/21
        "2324": "SP1", # La Liga 2023/24
        "2223": "SP1", # La Liga 2022/23
    }
    
    for season, league in seasons.items():
        output = f"data/raw/{league}_{season}.csv"
        download_season(league, season, output)

if __name__ == "__main__":
    main()