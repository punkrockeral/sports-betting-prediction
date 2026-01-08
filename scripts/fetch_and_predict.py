# scripts/fetch_and_predict.py
"""
Sistema de predicción diaria usando The Odds API.
- Premier League, Champions League, Europa League, Copa del Rey
- Cuotas en tiempo real (Pinnacle, Bet365)
- Modelo XGBoost entrenado con datos históricos
- Alertas por Telegram para EV > 2%
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, timezone, timedelta

def load_model():
    """Carga el modelo XGBoost entrenado."""
    model_path = "models/xgb_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modelo no encontrado: {model_path}")
    return joblib.load(model_path)

def get_team_stats_from_historical_data(team_name, historical_df, n_last=5):
    """Calcula estadísticas del equipo usando el dataset histórico existente."""
    if historical_df.empty:
        return {'goals_avg': 1.2, 'conceded_avg': 1.3, 'win_rate': 0.4}
    
    home_matches = historical_df[historical_df['HomeTeam'] == team_name].tail(n_last)
    away_matches = historical_df[historical_df['AwayTeam'] == team_name].tail(n_last)
    
    goals_scored = (home_matches['FTHG'].sum() + away_matches['FTAG'].sum())
    goals_conceded = (home_matches['FTAG'].sum() + away_matches['FTHG'].sum())
    wins = (len(home_matches[home_matches['FTR'] == 'H']) + len(away_matches[away_matches['FTR'] == 'A']))
    total_matches = len(home_matches) + len(away_matches)
    
    if total_matches == 0:
        return {'goals_avg': 1.2, 'conceded_avg': 1.3, 'win_rate': 0.4}
    
    return {
        'goals_avg': round(goals_scored / total_matches, 3),
        'conceded_avg': round(goals_conceded / total_matches, 3),
        'win_rate': round(wins / total_matches, 3)
    }

def send_telegram_alert(predictions):
    """Envía alertas para apuestas con EV > 2%."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return
    
    high_ev = [p for p in predictions if p.get("expected_value", 0) > 0.02]
    if not high_ev:
        return
    
    message = "🔥 *Nuevas apuestas con valor!*\n\n"
    for p in high_ev:
        ev_pct = p['expected_value'] * 100
        message += f"• {p['home_team']} vs {p['away_team']}\n"
        message += f"  → {p['predicted_outcome']} @{p['home_odds']:.2f} (EV: {ev_pct:.1f}%)\n\n"
    
    try:
        # ✅ URL CORREGIDA: sin espacios
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        print("✅ Alerta Telegram enviada")
    except Exception as e:
        print(f"❌ Error Telegram: {e}")

def fetch_real_matches():
    """Obtiene partidos de Premier League, Champions, Europa League y Copa del Rey."""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("⚠️  ODDS_API_KEY no configurada.")
        return []
    
    # ✅ COMPETICIONES SOLICITADAS
    sports = [
        "soccer_epl",                    # Premier League
        "soccer_uefa_champs_league",     # Champions League
        "soccer_uefa_europa_league",     # Europa League
        "soccer_spain_copa_del_rey"      # Copa del Rey
    ]
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    all_matches = []
    
    for sport in sports:
        try:
            # ✅ URL CORREGIDA: sin espacios
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            params = {
                "apiKey": api_key,
                "regions": "eu,us",
                "markets": "h2h",
                "oddsFormat": "decimal",
                "bookmakers": "pinnacle,bet365"
            }
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code != 200:
                continue
            
            odds_data = resp.json()
            # ✅ Corrección crítica: odds_data, no odds_
            for match in odds_:
                try:
                    match_time = datetime.fromisoformat(match["commence_time"].replace("Z", "+00:00"))
                    match_date = match_time.strftime("%Y-%m-%d")
                    
                    if match_date not in [today, tomorrow]:
                        continue
                    
                    home_odds = away_odds = draw_odds = None
                    for bookmaker in match.get("bookmakers", []):
                        if bookmaker["key"] in ["pinnacle", "bet365"]:
                            for market in bookmaker.get("markets", []):
                                if market["key"] == "h2h":
                                    outcomes = market["outcomes"]
                                    if len(outcomes) >= 3:
                                        home_odds = outcomes[0]["price"]
                                        away_odds = outcomes[1]["price"]
                                        draw_odds = outcomes[2]["price"]
                                    elif len(outcomes) == 2:  # Sin empate (raro en estas copas)
                                        home_odds = outcomes[0]["price"]
                                        away_odds = outcomes[1]["price"]
                                        draw_odds = 3.0
                                    break
                            if home_odds and away_odds:
                                break
                    
                    if home_odds and away_odds:
                        all_matches.append({
                            "match_id": match["id"],
                            "home_team": match["home_team"],
                            "away_team": match["away_team"],
                            "match_date": match["commence_time"],
                            "home_odds": float(home_odds),
                            "draw_odds": float(draw_odds) if draw_odds else 3.0,
                            "away_odds": float(away_odds)
                        })
                except:
                    continue
        except:
            continue
    
    print(f"✅ The Odds API: {len(all_matches)} partidos encontrados")
    return all_matches

def simulate_matches():
    """Datos simulados como respaldo."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return [{
        "match_id": "sim_1",
        "home_team": "Team A",
        "away_team": "Team B",
        "match_date": f"{today}T19:00:00Z",
        "home_odds": 2.10,
        "draw_odds": 3.40,
        "away_odds": 3.20
    }]

def predict_match(model, features):
    """Predice probabilidades usando el modelo XGBoost."""
    X = np.array([[
        features['home_goals_avg'],
        features['home_conceded_avg'],
        features['home_win_rate'],
        features['away_goals_avg'],
        features['away_conceded_avg'],
        features['away_win_rate']
    ]])
    
    try:
        probabilities = model.predict_proba(X)[0]
    except:
        pred = model.predict(X)[0]
        probabilities = [0, 0, 0]
        probabilities[pred] = 1.0
    
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]
    return probabilities

def calculate_ev(probability, odds):
    """Calcula el valor esperado (EV)."""
    return (probability * odds) - 1

def main():
    print("🎯 Iniciando predicción con The Odds API (Premier, UCL, UEL, Copa del Rey)...")
    
    # Cargar dataset histórico existente
    historical_path = "data/processed/football_matches_with_odds.csv"
    historical_df = pd.read_csv(historical_path) if os.path.exists(historical_path) else pd.DataFrame()
    
    if not historical_df.empty:
        try:
            historical_df['Date'] = pd.to_datetime(historical_df['Date'], format='%d/%m/%Y')
        except:
            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        historical_df = historical_df.sort_values('Date').reset_index(drop=True)
    
    # Obtener partidos reales
    matches = fetch_real_matches()
    if not matches:
        print("ℹ️  No hay partidos en competiciones europeas/hoy. Usando simulados.")
        matches = simulate_matches()
    
    # Cargar modelo y generar predicciones
    model = load_model()
    predictions = []
    
    for match in matches:
        home_stats = get_team_stats_from_historical_data(match["home_team"], historical_df)
        away_stats = get_team_stats_from_historical_data(match["away_team"], historical_df)
        
        features = {
            'home_goals_avg': home_stats["goals_avg"],
            'home_conceded_avg': home_stats["conceded_avg"],
            'home_win_rate': home_stats["win_rate"],
            'away_goals_avg': away_stats["goals_avg"],
            'away_conceded_avg': away_stats["conceded_avg"],
            'away_win_rate': away_stats["win_rate"]
        }
        
        probs = predict_match(model, features)
        prob_home, prob_draw, prob_away = probs
        
        ev_home = calculate_ev(prob_home, match["home_odds"])
        ev_draw = calculate_ev(prob_draw, match["draw_odds"])
        ev_away = calculate_ev(prob_away, match["away_odds"])
        
        evs = {"HOME": ev_home, "DRAW": ev_draw, "AWAY": ev_away}
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
    
    # Guardar predicciones
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    os.makedirs("data/predictions", exist_ok=True)
    output_path = f"data/predictions/{today}.json"
    
    # ✅ Serializador seguro para NumPy
    def numpy_converter(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "date": today,
            "model_used": "xgb_model.pkl",
            "predictions": predictions
        }, f, indent=2, default=numpy_converter)
    
    print(f"\n✅ Predicciones guardadas en {output_path}")
    send_telegram_alert(predictions)

if __name__ == "__main__":
    main()