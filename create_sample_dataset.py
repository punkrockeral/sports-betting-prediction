# create_sample_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Equipos reales de Premier League
teams = [
    "Manchester United", "Liverpool", "Chelsea", "Arsenal", "Manchester City",
    "Tottenham", "Leicester", "West Ham", "Everton", "Aston Villa",
    "Newcastle", "Wolves", "Crystal Palace", "Brighton", "Southampton",
    "Burnley", "Watford", "Norwich", "Brentford", "Leeds"
]

np.random.seed(42)
n_matches = 1000

data = []
start_date = datetime(2020, 8, 1)

for i in range(n_matches):
    date = start_date + timedelta(days=np.random.randint(0, 1460))
    home = np.random.choice(teams)
    away = np.random.choice([t for t in teams if t != home])
    
    # Simular goles (distribución realista)
    home_goals = np.random.poisson(1.4)
    away_goals = np.random.poisson(1.1)
    
    # Resultado
    if home_goals > away_goals:
        result = 'H'
    elif home_goals < away_goals:
        result = 'A'
    else:
        result = 'D'
    
    # Cuotas realistas (basadas en datos históricos)
    if result == 'H':
        b365h = round(np.random.uniform(1.8, 2.5), 2)
        b365d = round(np.random.uniform(3.2, 4.0), 2)
        b365a = round(np.random.uniform(3.5, 5.0), 2)
    elif result == 'A':
        b365h = round(np.random.uniform(3.5, 5.0), 2)
        b365d = round(np.random.uniform(3.2, 4.0), 2)
        b365a = round(np.random.uniform(1.8, 2.5), 2)
    else:
        b365h = round(np.random.uniform(2.8, 3.8), 2)
        b365d = round(np.random.uniform(3.0, 3.6), 2)
        b365a = round(np.random.uniform(2.8, 3.8), 2)
    
    data.append({
        "Div": "E0",
        "Date": date.strftime("%d/%m/%Y"),
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": home_goals,
        "FTAG": away_goals,
        "FTR": result,
        "B365H": b365h,
        "B365D": b365d,
        "B365A": b365a
    })

df = pd.DataFrame(data)
df.to_csv("data/processed/football_matches_with_odds.csv", index=False)
print("✅ Dataset sintético generado: data/processed/football_matches_with_odds.csv")
print(f"📊 {len(df)} partidos creados.")