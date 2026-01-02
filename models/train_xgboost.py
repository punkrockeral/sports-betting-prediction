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
    
    # ✅ Features SIN cuotas (solo estadísticas históricas)
    feature_cols = [
        'home_goals_avg', 'home_conceded_avg', 'home_win_rate',
        'away_goals_avg', 'away_conceded_avg', 'away_win_rate'
    ]
    
    # Verificación adicional
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Las siguientes columnas faltan en el dataset: {missing_cols}")
    
    X = df[feature_cols]
    y = df['target']  # 0=Local, 1=Empate, 2=Visitante
    
    print(f"📊 Features utilizadas: {feature_cols}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("🚀 Entrenando modelo XGBoost con regularización...")
    model = XGBClassifier(
        # 🔑 Parámetros de regularización fuerte
        n_estimators=80,        # Reducido de 200 → menos sobreajuste
        max_depth=3,           # Reducido de 5 → árboles más simples
        learning_rate=0.05,    # Reducido de 0.1 → aprendizaje más lento = más robusto
        gamma=0.5,             # ✅ Mínima pérdida para hacer split (0.5+)
        min_child_weight=5,    # ✅ Mínimo peso de instancias en nodo hijo (3+)
        reg_alpha=1.0,         # ✅ Regularización L1 (1.0+)
        reg_lambda=1.5,        # ✅ Regularización L2 (1.0+)
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
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