# train_xgboost.py
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

def train_model():
    input_path = "data/processed/features.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"⚠️  Archivo no encontrado: {input_path}. Ejecuta primero prepare_dataset.py")
    
    df = pd.read_csv(input_path)
    print(f"📥 Cargados {len(df)} partidos para entrenamiento.")
    
    feature_cols = [
        'B365H', 'B365D', 'B365A',
        'home_goals_avg', 'home_conceded_avg', 'home_win_rate',
        'away_goals_avg', 'away_conceded_avg', 'away_win_rate'
    ]
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"📊 Features utilizadas: {feature_cols}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4, stratify=y
    )
    
    print("🚀 Entrenando modelo XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softmax',
        num_class=3
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Resultados del modelo:")
    print(f"   Precisión general: {accuracy:.2%}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Local', 'Empate', 'Visitante'])}")
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Modelo guardado en: {model_path}")
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance.to_csv("models/feature_importance.csv", index=False)
    print(f"📈 Importancia de features guardada en: models/feature_importance.csv")
    print(importance)

if __name__ == "__main__":
    train_model()