# core/market_analyzer.py
def detect_line_movement(current_odds, historical_odds):
    """
    Retorna si la cuota ha bajado significativamente (posible apuesta inteligente).
    """
    movement = {}
    for outcome in ["home", "draw", "away"]:
        old = historical_odds.get(outcome, current_odds[outcome])
        new = current_odds[outcome]
        change = (old - new) / old  # Cuota bajó → valor positivo
        movement[outcome] = change
    return movement