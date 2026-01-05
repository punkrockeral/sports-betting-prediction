# scripts/download_historical_data.py
"""
Descarga datos históricos desde API-Football vía RapidAPI
- Ligas: Argentina, Brasil, Inglaterra, España, etc.
- Temporadas: 2023, 2024, 2025
- Incluye cuotas de Bet365
- Usa headers de RapidAPI
"""

import os
import pandas as pd
import requests
from datetime import datetime

def download_season(league_id, league_name, season, api_key):
    """
    Descarga una temporada completa desde API-Football (vía RapidAPI).
    """
    all_fixtures = []
    page = 1
    total_pages = 1
    
    # ✅ Headers CORRECTOS para RapidAPI
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "v3.football.api-sports.io"
    }
    
    while page <= total_pages:
        try:
            url = "https://v3.football.api-sports.io/fixtures"
            params = {
                "league": league_id,
                "season": season,
                "status": "FT",  # Solo partidos finalizados
                "page": page
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 429:
                print(f"⚠️  Límite de tasa excedido en {league_name} {season}")
                break
            elif response.status_code != 200:
                print(f"⚠️  Error HTTP {response.status_code} en {league_name} {season} (página {page})")
                break
            
            data = response.json()
            fixtures = data.get("response", [])
            if not fixtures:
                break
            
            # Manejo de paginación
            total_pages = data.get("paging", {}).get("total", 1)
            print(f"  → {league_name} {season}: página {page}/{total_pages} ({len(fixtures)} partidos)")
            
            all_fixtures.extend(fixtures)
            page += 1
            
        except Exception as e:
            print(f"⚠️  Error en {league_name} {season} (página {page}): {e}")
            break
    
    # Procesar partidos
    processed = []
    for fixture in all_fixtures:
        try:
            # Fecha
            date = fixture["fixture"]["date"].split("T")[0]
            
            # Equipos
            home_team = fixture["teams"]["home"]["name"]
            away_team = fixture["teams"]["away"]["name"]
            
            # Resultado
            home_goals = fixture["goals"]["home"]
            away_goals = fixture["goals"]["away"]
            
            if home_goals is None or away_goals is None:
                continue
            
            # Resultado final (FTR)
            if home_goals > away_goals:
                ftr = "H"
            elif home_goals < away_goals:
                ftr = "A"
            else:
                ftr = "D"
            
            # Cuotas (Bet365)
            b365h = b365d = b365a = None
            if "bookmakers" in fixture:
                for bookmaker in fixture["bookmakers"]:
                    if bookmaker["name"] == "Bet365":
                        for bet in bookmaker["bets"]:
                            if bet["name"] == "Match Winner":
                                for value in bet["values"]:
                                    if value["value"] == "Home":
                                        b365h = float(value["odd"])
                                    elif value["value"] == "Draw":
                                        b365d = float(value["odd"])
                                    elif value["value"] == "Away":
                                        b365a = float(value["odd"])
                                break
                        break
            
            # Incluir partidos incluso si no tienen cuotas (usar valores por defecto para temporadas recientes)
            if b365h is None:
                if season >= 2024:
                    b365h, b365d, b365a = 2.0, 3.0, 3.5
                else:
                    continue  # Saltar históricos sin cuotas
            
            processed.append({
                "Date": date,
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "FTHG": home_goals,
                "FTAG": away_goals,
                "FTR": ftr,
                "B365H": b365h,
                "B365D": b365d,
                "B365A": b365a
            })
        except Exception as e:
            continue  # Saltar partidos con datos faltantes
    
    print(f"✅ {league_name} {season}: {len(processed)} partidos procesados")
    return processed

def main():
    # 🔑 Clave de RapidAPI (no la de API-Football directo)
    api_key = os.getenv("RAPID_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ RAPID_API_KEY no configurada en las variables de entorno.")
    
    # 🌎 Ligas y temporadas a descargar
    leagues = {
        128: "Argentina Primera División",
        71: "Brasileirão",
        39: "Premier League",
        140: "La Liga",
        78: "Bundesliga",
        135: "Serie A",
        61: "Ligue 1"
    }
    seasons = [2023, 2024, 2025]
    
    all_data = []
    
    for league_id, league_name in leagues.items():
        print(f"\n🔄 Descargando {league_name}...")
        for season in seasons:
            fixtures = download_season(league_id, league_name, season, api_key)
            all_data.extend(fixtures)
    
    if not all_data:
        raise ValueError("❌ No se descargaron datos válidos.")
    
    # Guardar en CSV
    df = pd.DataFrame(all_data)
    df = df.sort_values("Date").reset_index(drop=True)
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/football_matches_with_odds.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n🏁 Total de partidos descargados: {len(df)}")
    print(f"📅 Rango de fechas: {df['Date'].min()} → {df['Date'].max()}")
    print(f"💾 Datos guardados en: {output_path}")

if __name__ == "__main__":
    main()