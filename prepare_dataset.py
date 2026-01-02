# prepare_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
    """Carga y limpia el dataset base."""
    df = pd.read_csv(filepath)
    
    # Convertir fecha a datetime (maneja múltiples formatos)
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    except:
        df['Date'] = pd.to_datetime(df['Date'])  # Intento genérico
    
    # Ordenar por fecha
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def calculate_team_stats(df, team_col, opponent_col, goals_for_col, goals_against_col, n_last=5):
    """
    Calcula estadísticas de rendimiento reciente para cada equipo.
    """
    teams = set(df[team_col].unique()) | set(df[opponent_col].unique())
    stats = {team: {
        'games': [],
        'goals_scored': [],
        'goals_conceded': [],
        'wins': []
    } for team in teams}

    home_features = []
    away_features = []

    for idx, row in df.iterrows():
        home_team = row[team_col]
        away_team = row[opponent_col]
        
        # Recuperar historial reciente
        home_last = stats[home_team]
        away_last = stats[away_team]
        
        # Calcular promedios (manejar caso sin historial)
        home_goals_avg = np.mean(home_last['goals_scored'][-n_last:]) if home_last['goals_scored'] else 1.4
        home_conceded_avg = np.mean(home_last['goals_conceded'][-n_last:]) if home_last['goals_conceded'] else 1.1
        home_win_rate = np.mean(home_last['wins'][-n_last:]) if home_last['wins'] else 0.4
        
        away_goals_avg = np.mean(away_last['goals_scored'][-n_last:]) if away_last['goals_scored'] else 1.1
        away_conceded_avg = np.mean(away_last['goals_conceded'][-n_last:]) if away_last['goals_conceded'] else 1.4
        away_win_rate = np.mean(away_last['wins'][-n_last:]) if away_last['wins'] else 0.3

        # Guardar features para este partido
        home_features.append({
            'home_goals_avg': home_goals_avg,
            'home_conceded_avg': home_conceded_avg,
            'home_win_rate': home_win_rate
        })
        
        away_features.append({
            'away_goals_avg': away_goals_avg,
            'away_conceded_avg': away_conceded_avg,
            'away_win_rate': away_win_rate
        })
        
        # Actualizar historial con el resultado ACTUAL
        home_goals = row[goals_for_col]
        away_goals = row[goals_against_col]
        
        # Local
        stats[home_team]['games'].append(idx)
        stats[home_team]['goals_scored'].append(home_goals)
        stats[home_team]['goals_conceded'].append(away_goals)
        stats[home_team]['wins'].append(1 if home_goals > away_goals else 0)
        
        # Visitante
        stats[away_team]['games'].append(idx)
        stats[away_team]['goals_scored'].append(away_goals)
        stats[away_team]['goals_conceded'].append(home_goals)
        stats[away_team]['wins'].append(1 if away_goals > home_goals else 0)
    
    return pd.DataFrame(home_features), pd.DataFrame(away_features)

def prepare_features(input_csv, output_csv):
    """Pipeline completo de preparación de features."""
    print("📥 Cargando datos...")
    df = load_data(input_csv)
    
    print("⚙️  Calculando estadísticas de equipos...")
    home_feat, away_feat = calculate_team_stats(
        df, 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', n_last=5
    )
    
    print("📊 Creando conjunto final de features...")
    # ✅ CORRECCIÓN: NO incluir cuotas en las features de entrenamiento
    features = pd.concat([
        df[['Date', 'HomeTeam', 'AwayTeam', 'FTR']],  # ← Sin B365H, B365D, B365A
        home_feat,
        away_feat
    ], axis=1)
    
    # Codificar el objetivo
    target_map = {'H': 0, 'D': 1, 'A': 2}
    features['target'] = features['FTR'].map(target_map)
    
    # Guardar
    features.to_csv(output_csv, index=False)
    print(f"✅ Features guardadas en {output_csv}")
    print(f"📈 Tamaño del dataset: {len(features)} partidos")

if __name__ == "__main__":
    prepare_features(
        input_csv="data/processed/football_matches_with_odds.csv",
        output_csv="data/processed/features.csv"
    )