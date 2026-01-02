# scripts/combine_historical_data.py
import pandas as pd
import glob
import os

os.makedirs("data/processed", exist_ok=True)

# Buscar todos los CSV en data/raw
files = glob.glob("data/raw/E0_*.csv")
if not files:
    raise FileNotFoundError("❌ No se encontraron archivos en data/raw/")

print(f"📂 Encontrados {len(files)} archivos: {files}")

# Combinar solo columnas necesarias
required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
df_list = []

for file in files:
    try:
        df = pd.read_csv(file)
        # Filtrar filas con cuotas completas
        df = df.dropna(subset=required_cols)
        df_list.append(df[required_cols])
        print(f"✅ Cargado: {file} → {len(df)} partidos")
    except Exception as e:
        print(f"⚠️  Error al cargar {file}: {e}")

if not df_list:
    raise ValueError("❌ Ningún archivo válido encontrado")

# Combinar y guardar
combined = pd.concat(df_list, ignore_index=True)
combined.to_csv("data/processed/football_matches_with_odds.csv", index=False)
print(f"\n✅ ¡Listo! Dataset combinado con {len(combined)} partidos reales.")