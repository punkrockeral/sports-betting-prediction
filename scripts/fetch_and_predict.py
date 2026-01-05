# scripts/fetch_and_predict.py
"""
Predicción diaria enfocada en COPAS INTERNACIONALES:
- Copa del Rey (España)
- FA Cup (Inglaterra)
- Copa Libertadores
- Copa Sudamericana
- Copa do Brasil

No incluye ligas nacionales (ej: Liga Profesional Argentina).
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
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        print("✅ Alerta Telegram enviada")
    except Exception as e:
        print(f"❌ Error Telegram: {e}")

def fetch_api_football_fixtures():
    """
    Obtiene partidos FUTUROS de COPAS INTERNACIONALES activas en enero-febrero 2026.
    """
    api_key = os.getenv("RAPID_API_KEY")
    if not api_key:
        print("⚠️  RAPID_API_KEY no configurada.")
        return []
    
    # 🔥 COPAS RECOMENDADAS (sin ligas nacionales)
    copas = [
        143,  # Copa del Rey (España) - ENERO
        45,   # FA Cup (Inglaterra) - ENERO
        133,  # Copa Libertadores - FEBRERO
        134,  # Copa Sudamericana - FEBRERO
        135   # Copa do Brasil - FEBRERO
    ]
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    dates = [today, tomorrow]
    
    all_matches = []
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "v3.football.api-sports.io"
    }
    
    print(f"📡 Consultando copas internacionales para {today} y {tomorrow}...")
    
    for copa_id in copas:
        for date in dates:
            try:
                url = "https://v3.football.api-sports.io/fixtures"
                params = {
                    "league": copa_id,
                    "date": date,
                    "timezone": "UTC"
                }
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                
                if resp.status_code != 200:
                    continue
                
                data = resp.json()
                fixtures = data.get("response", [])
                
                for fixture in fixtures:
                    # Solo partidos no iniciados
                    if fixture["fixture"]["status"]["short"] != "NS":
                        continue
                    
                    # Extraer cuotas de Bet365
                    home_odds = draw_odds = away_odds = None
                    if fixture.get("bookmakers"):
                        for bookmaker in fixture["bookmakers"]:
                            if bookmaker["name"] == "Bet365":
                                for bet in bookmaker["bets"]:
                                    if bet["name"] == "Match Winner":
                                        for value in bet["values"]:
                                            if value["value"] == "Home":
                                                home_odds = float(value["odd"])
                                            elif value["value"] == "Draw":
                                                draw_odds = float(value["odd"])
                                            elif value["value"] == "Away":
                                                away_odds = float(value["odd"])
                                        break
                                break
                    
                    if home_odds and draw_odds and away_odds:
                        all_matches.append({
                            "match_id": fixture["fixture"]["id"],
                            "home_team": fixture["teams"]["home"]["name"],
                            "away_team": fixture["teams"]["away"]["name"],
                            "match_date": fixture["fixture"]["date"],
                            "home_odds": home_odds,
                            "draw_odds": draw_odds,
                            "away_odds": away_odds
                        })
            except Exception as e:
                continue  # Saltar errores silenciosamente
    
    print(f"✅ Copas internacionales: {len(all_matches)} partidos encontrados")
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
    print("🎯 Iniciando predicción para COPAS INTERNACIONALES...")
    
    # Cargar dataset histórico existente (football-data.co.uk)
    historical_path = "data/processed/football_matches_with_odds.csv"
    historical_df = pd.read_csv(historical_path) if os.path.exists(historical_path) else pd.DataFrame()
    
    if not historical_df.empty:
        try:
            historical_df['Date'] = pd.to_datetime(historical_df['Date'], format='%d/%m/%Y')
        except:
            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        historical_df = historical_df.sort_values('Date').reset_index(drop=True)
    
    # Obtener partidos reales de copas
    matches = fetch_api_football_fixtures()
    if not matches:
        print("ℹ️  No hay partidos en copas internacionales. Usando simulados.")
        matches = simulate_matches()
    
    # Cargar modelo y generar predicciones
    model = load_model()
    predictions = []
    
    for match in matches:
        # Calcular estadísticas históricas
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
            "home_odds": match["home_odds"],
            "draw_odds": match["draw_odds"],
            "away_odds": match["away_odds"],
            "predicted_outcome": best_outcome,
            "expected_value": round(best_ev, 3),
            "prob_home": round(prob_home, 3),
            "prob_draw": round(prob_draw, 3),
            "prob_away": round(prob_away, 3)
        })
        print(f"  → {match['home_team']} vs {match['away_team']}: {best_outcome} (EV: {best_ev:.2%})")
    
    # Guardar predicciones
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    os.makedirs("data/predictions", exist_ok=True)
    output_path = f"data/predictions/{today}.json"
    
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "date": today,
            "model_used": "xgb_model.pkl",
            "predictions": predictions
        }, f, indent=2)
    
    print(f"\n✅ Predicciones guardadas en {output_path}")
    send_telegram_alert(predictions)

if __name__ == "__main__":
    main()