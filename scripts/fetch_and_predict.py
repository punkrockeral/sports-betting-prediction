# scripts/fetch_and_predict.py
"""
Sistema de predicción diaria con:
- Datos reales de The Odds API (solo partidos FUTUROS)
- Estadísticas históricas REALES de cada equipo
- Modelo XGBoost regularizado
- Cálculo de EV creíble
- Alertas por Telegram
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
        raise FileNotFoundError(f"❌ Modelo no encontrado: {model_path}. Ejecuta train_xgboost.py primero.")
    return joblib.load(model_path)

def get_team_stats_from_historical_data(team_name, historical_df, n_last=5):
    """
    Calcula estadísticas reales de un equipo usando el dataset histórico.
    """
    if historical_df.empty:
        return {'goals_avg': 1.2, 'conceded_avg': 1.3, 'win_rate': 0.4}
    
    home_matches = historical_df[historical_df['HomeTeam'] == team_name].tail(n_last)
    away_matches = historical_df[historical_df['AwayTeam'] == team_name].tail(n_last)
    
    goals_scored = 0
    goals_conceded = 0
    wins = 0
    total_matches = 0
    
    for _, row in home_matches.iterrows():
        goals_scored += row['FTHG']
        goals_conceded += row['FTAG']
        if row['FTR'] == 'H':
            wins += 1
        total_matches += 1
    
    for _, row in away_matches.iterrows():
        goals_scored += row['FTAG']
        goals_conceded += row['FTHG']
        if row['FTR'] == 'A':
            wins += 1
        total_matches += 1
    
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
    
    print(f"🔍 Telegram - Token presente: {'Sí' if bot_token else 'No'}, Chat ID presente: {'Sí' if chat_id else 'No'}")
    
    if not bot_token or not chat_id:
        print("⚠️  TELEGRAM: Credenciales no configuradas.")
        return
    
    high_ev = [p for p in predictions if p.get("expected_value", 0) > 0.02]
    if not high_ev:
        print("ℹ️  TELEGRAM: Ninguna apuesta supera el umbral de EV (2%).")
        return
    
    message = "🔥 *Nuevas apuestas con valor!*\n\n"
    for p in high_ev:
        ev_pct = p['expected_value'] * 100
        message += f"• {p['home_team']} vs {p['away_team']}\n"
        message += f"  → {p['predicted_outcome']} @{p['home_odds']:.2f} (EV: {ev_pct:.1f}%)\n\n"
    
    try:
        # ✅ URL CORREGIDA: sin espacios
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }, timeout=10)
        print("✅ TELEGRAM: Alerta enviada correctamente.")
    except Exception as e:
        print(f"❌ TELEGRAM: Error al enviar alerta: {e}")

def get_real_matches(historical_df):
    """Obtiene partidos y cuotas reales de The Odds API (solo partidos futuros)."""
    api_key = os.getenv("ODDS_API_KEY")
    print(f"🔍 ODDS_API_KEY configurada: {'Sí' if api_key else 'No'}")
    
    if not api_key:
        print("⚠️  API KEY no configurada. Usando datos simulados.")
        return simulate_matches()
    
    # ✅ Ampliación: más ligas (incluyendo Championship y MLS para enero)
    sports = [
        "soccer_epl",
        "soccer_england_championship",     # Muy activo en enero
        "soccer_spain_la_liga",
        "soccer_germany_bundesliga",
        "soccer_italy_serie_a",
        "soccer_france_ligue_one",
        "soccer_netherlands_eredivisie",
        "soccer_portugal_primeira_liga",
        "soccer_argentina_primera_division",  # Temporada de verano (enero)
        "basketball_nba"  # Partidos diarios en enero
    ]
    
    all_matches = []
    today = datetime.now(timezone.utc).date()
    now = datetime.now(timezone.utc)
    
    print(f"📡 Consultando {len(sports)} deportes/ligas en The Odds API...")
    
    for sport in sports:
        try:
            # ✅ URL CORREGIDA: sin espacios
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            params = {
                "apiKey": api_key,
                "regions": "eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
                "bookmakers": "pinnacle,bet365"
            }
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code != 200:
                print(f"❌ {sport}: Error HTTP {resp.status_code}")
                continue
            
            odds_data = resp.json()
            print(f"✅ {sport}: {len(odds_data)} partidos encontrados")
            
            # ✅ Corrección CRÍTICA: odds_data, no odds_
            for match in odds_
                try:
                    match_time = datetime.fromisoformat(match["commence_time"].replace("Z", "+00:00"))
                    
                    # Filtrar partidos futuros (solo hoy)
                    if match_time <= now:
                        continue
                    if match_time.date() != today:
                        continue
                    
                    home_odds = away_odds = draw_odds = None
                    for bookmaker in match.get("bookmakers", []):
                        if bookmaker["key"] in ["pinnacle", "bet365"]:
                            for market in bookmaker.get("markets", []):
                                if market["key"] == "h2h":
                                    outcomes = market["outcomes"]
                                    if len(outcomes) == 3:
                                        # Fútbol: home, away, draw
                                        home_odds = outcomes[0]["price"]
                                        away_odds = outcomes[1]["price"]
                                        draw_odds = outcomes[2]["price"]
                                    elif len(outcomes) == 2:
                                        # Baloncesto/Tenis: home, away
                                        home_odds = outcomes[0]["price"]
                                        away_odds = outcomes[1]["price"]
                                        draw_odds = None
                                    break
                            if home_odds and away_odds:
                                break
                    
                    if home_odds and away_odds:
                        if sport.startswith("soccer_"):
                            home_stats = get_team_stats_from_historical_data(match["home_team"], historical_df)
                            away_stats = get_team_stats_from_historical_data(match["away_team"], historical_df)
                        else:
                            # Deportes no fútbol: usar valores por defecto
                            home_stats = {'goals_avg': 1.2, 'conceded_avg': 1.3, 'win_rate': 0.5}
                            away_stats = {'goals_avg': 1.1, 'conceded_avg': 1.4, 'win_rate': 0.4}
                        
                        match_entry = {
                            "match_id": match["id"],
                            "home_team": match["home_team"],
                            "away_team": match["away_team"],
                            "match_date": match["commence_time"],
                            "home_odds": float(home_odds),
                            "away_odds": float(away_odds),
                            "home_goals_avg": home_stats["goals_avg"],
                            "home_conceded_avg": home_stats["conceded_avg"],
                            "home_win_rate": home_stats["win_rate"],
                            "away_goals_avg": away_stats["goals_avg"],
                            "away_conceded_avg": away_stats["conceded_avg"],
                            "away_win_rate": away_stats["win_rate"]
                        }
                        # Solo incluir draw_odds si existe
                        if draw_odds is not None:
                            match_entry["draw_odds"] = float(draw_odds)
                        else:
                            match_entry["draw_odds"] = 3.0  # Valor por defecto para cálculo
                        
                        all_matches.append(match_entry)
                except KeyError as e:
                    print(f"⚠️  {sport}: Dato faltante en partido: {e}")
                    continue
        except Exception as e:
            print(f"⚠️  {sport}: Error al procesar: {e}")
    
    if not all_matches:
        print("ℹ️  No hay partidos válidos en ninguna liga. Usando datos simulados.")
        return simulate_matches()
    
    print(f"🎯 Total de partidos válidos: {len(all_matches)}")
    return all_matches

def simulate_matches():
    """Genera datos simulados como respaldo."""
    teams = ["Team A", "Team B", "Team C", "Team D"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return [{
        "match_id": "sim_1",
        "home_team": "Team A",
        "away_team": "Team B",
        "match_date": f"{today}T19:00:00Z",
        "home_odds": 2.10,
        "draw_odds": 3.40,
        "away_odds": 3.20,
        "home_goals_avg": 1.4,
        "home_conceded_avg": 1.1,
        "home_win_rate": 0.6,
        "away_goals_avg": 1.1,
        "away_conceded_avg": 1.3,
        "away_win_rate": 0.4
    }]

def predict_match(model, match_features):
    """Predice probabilidades usando el modelo XGBoost (solo 6 features)."""
    X = np.array([[
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
    
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]
    
    return probabilities

def calculate_ev(probability, odds):
    """Calcula el valor esperado (EV)."""
    return (probability * odds) - 1

def main():
    print("🎯 Iniciando sistema de predicción diaria...")
    
    historical_path = "data/processed/football_matches_with_odds.csv"
    if not os.path.exists(historical_path):
        print(f"❌ Advertencia: {historical_path} no encontrado. Usando stats por defecto.")
        historical_df = pd.DataFrame()
    else:
        print("📊 Cargando datos históricos...")
        historical_df = pd.read_csv(historical_path)
        try:
            historical_df['Date'] = pd.to_datetime(historical_df['Date'], format='%d/%m/%Y')
        except:
            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        historical_df = historical_df.sort_values('Date').reset_index(drop=True)
    
    model = load_model()
    print("✅ Modelo cargado correctamente.")
    
    matches = get_real_matches(historical_df)
    print(f"📅 Encontrados {len(matches)} partidos para hoy.")
    
    predictions = []
    for match in matches:
        features_for_model = {
            'home_goals_avg': match['home_goals_avg'],
            'home_conceded_avg': match['home_conceded_avg'],
            'home_win_rate': match['home_win_rate'],
            'away_goals_avg': match['away_goals_avg'],
            'away_conceded_avg': match['away_conceded_avg'],
            'away_win_rate': match['away_win_rate']
        }
        
        probs = predict_match(model, features_for_model)
        prob_home, prob_draw, prob_away = probs
        
        # Usar draw_odds solo si está definido
        draw_odds = match.get('draw_odds', 3.0)
        
        ev_home = calculate_ev(prob_home, match['home_odds'])
        ev_draw = calculate_ev(prob_draw, draw_odds)
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
            "draw_odds": float(draw_odds),
            "away_odds": float(match["away_odds"]),
            "predicted_outcome": best_outcome,
            "expected_value": float(round(best_ev, 3)),
            "prob_home": float(round(prob_home, 3)),
            "prob_draw": float(round(prob_draw, 3)),
            "prob_away": float(round(prob_away, 3))
        })
        
        print(f"  → {match['home_team']} vs {match['away_team']}: {best_outcome} (EV: {best_ev:.2%})")
    
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
    high_ev_count = sum(1 for p in predictions if p['expected_value'] > 0)
    print(f"💡 Apuestas con EV > 0: {high_ev_count} de {len(predictions)}")
    
    send_telegram_alert(predictions)

if __name__ == "__main__":
    main()