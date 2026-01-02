# scripts/combine_historical_data.py
import pandas as pd
import glob
import os

def main():
    # Crear directorios si no existen
    os.makedirs("data/processed", exist_ok=True)
    
    # Buscar todos los CSV en data/raw
    files = glob.glob("data/raw/*.csv")
    if not files:
        raise FileNotFoundError("❌ No se encontraron archivos en data/raw/")
    
    # Columnas requeridas (algunos archivos pueden no tener todas)
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    odds_cols = ['B365H', 'B365D', 'B365A']  # Opcionales (no todos los archivos las tienen)
    
    df_list = []
    
    for file in files:
        try:
            df = pd.read_csv(file)
            
            # Verificar columnas obligatorias
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️  Advertencia: {file} no tiene todas las columnas obligatorias. Omitiendo.")
                continue
            
            # Asegurar que las columnas de cuotas existan (llenar con NaN si no)
            for col in odds_cols:
                if col not in df.columns:
                    df[col] = pd.NA
            
            # Seleccionar solo las columnas necesarias
            selected_cols = required_cols + odds_cols
            df_list.append(df[selected_cols])
            print(f"✅ Procesado: {file} → {len(df)} partidos")
            
        except Exception as e:
            print(f"⚠️  Error al procesar {file}: {e}")
    
    if not df_list:
        raise ValueError("❌ Ningún archivo válido con columnas obligatorias encontrado")
    
    # Combinar todos los DataFrames
    combined = pd.concat(df_list, ignore_index=True)
    
    # Ordenar por fecha (manejar múltiples formatos)
    try:
        combined['Date'] = pd.to_datetime(combined['Date'], format='%d/%m/%Y')
    except:
        combined['Date'] = pd.to_datetime(combined['Date'])  # Intento genérico
    
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    # Guardar resultado
    output_path = "data/processed/football_matches_with_odds.csv"
    combined.to_csv(output_path, index=False)
    print(f"\n✅ Dataset combinado guardado en {output_path}")
    print(f"📊 Total de partidos: {len(combined)}")
    print(f"📅 Rango de fechas: {combined['Date'].min()} → {combined['Date'].max()}")

if __name__ == "__main__":
    main()