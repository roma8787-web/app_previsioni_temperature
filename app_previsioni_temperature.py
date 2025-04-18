
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from datetime import timedelta

st.set_page_config(page_title="📈 Previsioni delle Temperature – 7 Giorni", layout="wide")

st.title("📈 Previsioni delle Temperature – 7 Giorni")
st.markdown("Modello LSTM addestrato sui dati storici per prevedere l'andamento delle temperature.")

# 📥 Carica dati
try:
    df = pd.read_csv("Dati_temperatura.csv")
    df.columns = [col.lower() for col in df.columns]

    if 'data' not in df.columns or 'temperatura' not in df.columns:
        st.error("❌ Il file CSV deve contenere le colonne 'data' e 'temperatura'.")
        st.stop()

    df['Date'] = pd.to_datetime(df['data'])
    df['Temperature'] = df['temperatura']
    df = df.sort_values("Date")
except Exception as e:
    st.error(f"❌ Errore nel caricamento del CSV: {e}")
    st.stop()

# Carica scaler e modello
scaler = joblib.load("scaler.save")
model = tf.keras.models.load_model("modello_lstm.keras")

# Prepara input per previsione
data = df['Temperature'].values.reshape(-1, 1)
data_scaled = scaler.transform(data)
last_30_days = data_scaled[-30:]

# Previsioni
future_scaled = []
input_seq = last_30_days
for _ in range(7):
    prediction = model.predict(input_seq.reshape(1, 30, 1), verbose=0)
    future_scaled.append(prediction[0])
    input_seq = np.vstack((input_seq[1:], prediction))

future_scaled = np.array(future_scaled)
future = scaler.inverse_transform(future_scaled).flatten()

# Date future
last_date = df['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(7)]

# Mostra grafico
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Temperature'], label="Storico")
ax.plot(future_dates, future, label="Previsione", linestyle='--', marker='o')
ax.set_title("Previsioni Temperatura (7 Giorni)")
ax.set_xlabel("Data")
ax.set_ylabel("Temperatura")
ax.legend()
st.pyplot(fig)

# Tabella
forecast_df = pd.DataFrame({'Data': future_dates, 'Temperatura Prevista': future})
st.subheader("📋 Tabella Previsioni")
st.dataframe(forecast_df)
