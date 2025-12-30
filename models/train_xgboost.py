# models/train_xgboost.py
import pandas as pd
from xgboost import XGBClassifier
import joblib

# Cargar dataset histórico (CSV con +20k filas)
df = pd.read_csv("data/processed/football_matches.csv")

# Features: goles últimos 5, posesión, tiros, etc.
X = df[["home_goals_avg", "away_goals_avg", "home_wins", "away_losses", "odds_home"]]
y = df["result"]  # HOME, DRAW, AWAY

model = XGBClassifier(n_estimators=200, max_depth=5)
model.fit(X, y)

joblib.dump(model, "models/xgb_model.pkl")