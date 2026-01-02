# scripts/fetch_and_predict.py
import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

def load_model():
    model_path = "models/xgb_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modelo no encontrado: {model_path}. Ejecuta train_xgboost.py primero.")
    return joblib.load(model_path)

def simulate_todays_matches():
    teams = [
        "Manchester United", "Liverpool", "Chelsea", "Arsenal", "Manchester City",
        "Tottenham", "Leicester", "West Ham", "Everton", "Aston Villa"
    ]
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    matches = []
    
    for i in range(3):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        
        home_odds = round(np.random.uniform(1.8, 3.0), 2)
        draw_odds = round(np.random.uniform(3.2, 4.0), 2)
        away_odds = round(np.random.uniform(2.5, 4.5), 2)
        
        matches.append({
            "match_id": f"match_{i+1}",
            "home_team": home,
            "away_team": away,
            "match_date": f"{today}T19:00:00Z",
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds,
            "home_goals_avg": 1.4,
            "home_conceded_avg": 1.1,
            "home_win_rate": 0.6,
            "away_goals_avg": 1.1,
            "away_conceded_avg": 1.3,
            "away_win_rate": 0.4
        })
    
    return matches

def predict_match(model, match_features):
    feature_order = [
        'B365H', 'B365D', 'B365A',
        'home_goals_avg', 'home_conceded_avg', 'home_win_rate',
        'away_goals_avg', 'away_conceded_avg', 'away_win_rate'
    ]
    
    X = np.array([[
        match_features['home_odds'],
        match_features['draw_odds'],
        match_features['away_odds'],
        match_features['home_goals_avg'],
        match_features['home_conceded_avg'],
        match_features['home_win_rate'],
        match_features['away_goals_avg'],
        match_features['away_conceded_avg'],
        match_features['away_win_rate']
    ]])
    
    try:
        probabilities = model.predict_proba(X)[0]
    except:
        pred = model.predict(X)[0]
        probabilities = [0, 0, 0]
        probabilities[pred] = 1.0
    
    return probabilities

def calculate_ev(probability, odds):
    return (probability * odds) - 1

def main():
    print("🎯 Iniciando sistema de predicción diaria...")
    
    model = load_model()
    print("✅ Modelo cargado correctamente.")
    
    matches = simulate_todays_matches()
    print(f"📅 Encontrados {len(matches)} partidos para hoy.")
    
    predictions = []
    for match in matches:
        features = {
            'home_odds': match['home_odds'],
            'draw_odds': match['draw_odds'],
            'away_odds': match['away_odds'],
            'home_goals_avg': match['home_goals_avg'],
            'home_conceded_avg': match['home_conceded_avg'],
            'home_win_rate': match['home_win_rate'],
            'away_goals_avg': match['away_goals_avg'],
            'away_conceded_avg': match['away_conceded_avg'],
            'away_win_rate': match['away_win_rate']
        }
        
        probs = predict_match(model, features)
        prob_home, prob_draw, prob_away = probs
        
        ev_home = calculate_ev(prob_home, match['home_odds'])
        ev_draw = calculate_ev(prob_draw, match['draw_odds'])
        ev_away = calculate_ev(prob_away, match['away_odds'])
        
        evs = {'HOME': ev_home, 'DRAW': ev_draw, 'AWAY': ev_away}
        best_outcome = max(evs, key=evs.get)
        best_ev = evs[best_outcome]
        
        predictions.append({
            "match_id": match["match_id"],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "match_date": match["match_date"],
            "home_odds": float(match["home_odds"]),
            "draw_odds": float(match["draw_odds"]),
            "away_odds": float(match["away_odds"]),
            "predicted_outcome": best_outcome,
            "expected_value": float(round(best_ev, 3)),
            "prob_home": float(round(prob_home, 3)),
            "prob_draw": float(round(prob_draw, 3)),
            "prob_away": float(round(prob_away, 3))
        })
        
        print(f"  → {match['home_team']} vs {match['away_team']}: {best_outcome} (EV: {best_ev:.2%})")
    
    # Guardar resultados
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    os.makedirs("data/predictions", exist_ok=True)
    output_path = f"data/predictions/{today}.json"
    
    def default_serializer(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} not JSON serializable")
    
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "date": today,
            "model_used": "xgb_model.pkl",
            "predictions": predictions
        }, f, indent=2, default=default_serializer)
    
    print(f"\n✅ Predicciones guardadas en {output_path}")
    print(f"💡 Apuestas con EV > 0: {sum(1 for p in predictions if p['expected_value'] > 0)} de {len(predictions)}")

if __name__ == "__main__":
    main()