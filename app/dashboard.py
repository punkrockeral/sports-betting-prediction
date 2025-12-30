# app/dashboard.py
import streamlit as st
import json
import glob
from datetime import datetime

st.set_page_config(page_title="Betting Predictor", layout="wide")
st.title("🎯 Predicción de Apuestas Deportivas")

# Cargar todas las predicciones
pred_files = sorted(glob.glob("data/predictions/*.json"), reverse=True)
if not pred_files:
    st.warning("No hay predicciones aún.")
else:
    latest = pred_files[0]
    with open(latest) as f:
        data = json.load(f)
    
    st.subheader(f"Predicciones del {data['date']}")
    
    # Filtrar por EV > 0.1
    high_value = [p for p in data["predictions"] if p.get("expected_value", 0) > 0.1]
    
    if high_value:
        for p in high_value:
            col1, col2, col3, col4 = st.columns(4)
            col1.write(f"**{p['home_team']} vs {p['away_team']}**")
            col2.write(f"🔮 {p['predicted_outcome']}")
            col3.write(f"💰 EV: {p['expected_value']:.2%}")
            col4.write(f"📊 Cuotas: {p['home_odds']:.2f} | {p['draw_odds']:.2f} | {p['away_odds']:.2f}")
    else:
        st.info("No hay apuestas con valor esperado positivo hoy.")