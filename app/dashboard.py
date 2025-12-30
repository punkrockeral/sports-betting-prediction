# app/dashboard.py
import streamlit as st
import json
import glob
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Betting Predictor Pro", layout="wide")
st.title("🎯 Sistema Profesional de Predicción de Apuestas")

# Cargar todas las predicciones
pred_files = sorted(glob.glob("data/predictions/*.json"), reverse=True)
if not pred_files:
    st.warning("⚠️ No hay predicciones generadas aún.")
    st.stop()

# Selector de fecha
dates = [f.split("/")[-1].replace(".json", "") for f in pred_files]
selected_date = st.selectbox("📅 Selecciona una fecha", dates, index=0)

# Cargar datos de la fecha seleccionada
with open(f"data/predictions/{selected_date}.json") as f:
    data = json.load(f)

# Convertir a DataFrame
rows = []
for p in data.get("predictions", []):
    rows.append({
        "Deporte": p.get("sport", "Fútbol"),
        "Liga": p.get("league", "Desconocida"),
        "Partido": f"{p['home_team']} vs {p['away_team']}",
        "Predicción": p["predicted_outcome"],
        "EV": p.get("expected_value", 0),
        "Cuota": p.get("home_odds") if p["predicted_outcome"] == "HOME" else (
                  p.get("draw_odds") if p["predicted_outcome"] == "DRAW" else p.get("away_odds")),
        "Fecha": p.get("match_date", "")[:10]
    })

if not rows:
    st.info("ℹ️ No hay predicciones para esta fecha.")
    st.stop()

df = pd.DataFrame(rows)

# Filtros
col1, col2 = st.columns(2)
with col1:
    deportes = ["Todos"] + sorted(df["Deporte"].unique())
    filtro_deporte = st.selectbox("⚽ Filtrar por deporte", deportes)
with col2:
    min_ev = st.slider("📉 EV mínimo", 0.0, 0.5, 0.1, 0.01)

# Aplicar filtros
if filtro_deporte != "Todos":
    df = df[df["Deporte"] == filtro_deporte]
df = df[df["EV"] >= min_ev].sort_values("EV", ascending=False)

# Mostrar
st.subheader(f"✅ {len(df)} apuestas con valor (EV ≥ {min_ev:.0%})")
st.dataframe(
    df[["Deporte", "Liga", "Partido", "Predicción", "Cuota", "EV"]],
    column_config={
        "EV": st.column_config.NumberColumn("EV", format="%.2f%%", help="Valor esperado"),
        "Cuota": st.column_config.NumberColumn("Cuota", format="%.2f")
    },
    hide_index=True,
    use_container_width=True
)

# Botón de descarga
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Descargar como CSV",
    csv,
    f"apuestas_{selected_date}.csv",
    "text/csv"
)