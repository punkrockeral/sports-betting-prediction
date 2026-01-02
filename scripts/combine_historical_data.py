# scripts/combine_historical_data.py
import pandas as pd
import glob
import os

def main():
    # Asegurar que existen los directorios
    os.makedirs("data/processed", exist_ok=True)
    
    # Buscar todos los CSV descargados
    files = glob.glob("data/raw/*.csv")
    if not files:
        raise FileNotFoundError("❌ No se encontraron archivos en data/raw/")
    
    # Combinar solo columnas necesarias
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
    df_list = []
    
    for file in files:
        try:
            df = pd.read_csv(file)
            if all(col in df.columns for col in required_cols):
                df_list.append(df[required_cols])
                print(f"✅ Cargado: {file}")
        except Exception as e:
            print(f"⚠️  Error al cargar {file}: {e}")
    
    if not df_list:
        raise ValueError("❌ Ningún archivo válido encontrado")
    
    # Combinar y guardar
    combined = pd.concat(df_list, ignore_index=True)
    output_path = "data/processed/football_matches_with_odds.csv"
    combined.to_csv(output_path, index=False)
    print(f"✅ Dataset combinado guardado en {output_path} ({len(combined)} partidos)")

if __name__ == "__main__":
    main()