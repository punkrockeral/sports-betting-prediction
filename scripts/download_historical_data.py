# scripts/download_historical_data.py
"""
Descarga datos históricos de football-data.co.uk para múltiples ligas y temporadas.
Funciona en GitHub Actions (donde el sitio está accesible).
"""

import pandas as pd
import os
import time
import requests
from datetime import datetime

def download_with_retry(url, max_retries=3, delay=2):
    """Descarga un CSV con reintentos en caso de fallo."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Verificar que el contenido sea CSV (no página de error)
            if response.headers.get('content-type', '').startswith('text/csv') or \
               (response.text and 'Date,HomeTeam' in response.text[:100]):
                return pd.read_csv(url)
            else:
                print(f"⚠️  Contenido no CSV en {url} (intentos: {attempt + 1})")
        except Exception as e:
            print(f"⚠️  Intento {attempt + 1} fallido para {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    
    return None

def download_league(league_code, season, output_dir="data/raw"):
    """Descarga una temporada específica de una liga."""
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
    output_path = os.path.join(output_dir, f"{league_code}_{season}.csv")
    
    # Saltar si ya existe (útil para desarrollo local)
    if os.path.exists(output_path):
        print(f"⏭️  Ya existe: {output_path}")
        return True
    
    os.makedirs(output_dir, exist_ok=True)
    df = download_with_retry(url)
    
    if df is not None:
        # Guardar solo columnas necesarias para ahorrar espacio
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
        available_cols = [col for col in required_cols if col in df.columns]
        df[available_cols].to_csv(output_path, index=False)
        print(f"✅ Descargado: {league_code} {season} → {len(df)} partidos")
        return True
    else:
        print(f"❌ Falló la descarga de {league_code} {season}")
        return False

def main():
    # 🔥 Configuración: Ligas y temporadas (ajustable)
    config = {
        # Liga, Código, Temporadas
        "Premier League": ("E0", ["2324", "2223", "2122", "2021"]),
        "La Liga": ("SP1", ["2324", "2223", "2122"]),
        "Bundesliga": ("D1", ["2324", "2223"]),
        "Serie A": ("I1", ["2324", "2223"]),
        "Ligue 1": ("F1", ["2324", "2223"]),
        "Eredivisie": ("N1", ["2324", "2223"]),
        "Primeira Liga": ("P1", ["2324", "2223"]),
        "Championship": ("E1", ["2324", "2223"])
    }
    
    total_downloaded = 0
    start_time = datetime.now()
    
    for league_name, (code, seasons) in config.items():
        print(f"\n🔄 Procesando {league_name} ({code})...")
        for season in seasons:
            if download_league(code, season):
                total_downloaded += 1
    
    elapsed = datetime.now() - start_time
    print(f"\n🏁 Descarga completada en {elapsed.total_seconds():.1f} segundos")
    print(f"📊 Archivos descargados: {total_downloaded}")

if __name__ == "__main__":
    main()