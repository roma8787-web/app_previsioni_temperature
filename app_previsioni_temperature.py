import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title=" Previsioni delle Temperature", layout="wide")
st.title("Previsioni delle Temperature – 7 Giorni")
st.markdown("Modello LSTM addestrato sui dati storici per prevedere l'andamento delle temperature.")

# Caricamento dei dati
df = pd.read_csv("Dati_temperatura.csv")
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data')

# Visualizzazione dati
st.subheader("Andamento Storico")
st.line_chart(df.set_index('data')['temperatura'])

# Preprocessing per LSTM
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['temperatura']])

# Creazione sequenze
n_steps = 30
X, y = [], []
for i in range(n_steps, len(data_scaled)):
    X.append(data_scaled[i - n_steps:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Costruzione modello solo se non esiste già
if not os.path.exists("modello_lstm.h5"):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)
    model.save("modello_lstm.h5")
else:
    model = tf.keras.models.load_model("modello_lstm.h5")

# Previsione prossimi 7 giorni
last_sequence = data_scaled[-n_steps:]
predictions = []
for _ in range(7):
    input_seq = last_sequence.reshape((1, n_steps, 1))
    pred = model.predict(input_seq, verbose=0)[0][0]
    predictions.append(pred)
    last_sequence = np.append(last_sequence[1:], [[pred]], axis=0)

# Decodifica delle previsioni
predicted_temps = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
future_dates = pd.date_range(start=df['data'].max() + pd.Timedelta(days=1), periods=7)

# Visualizzazione
st.subheader("🌧️ Previsioni per i prossimi 7 giorni")
forecast_df = pd.DataFrame({"Data": future_dates, "Temperatura prevista": predicted_temps})
st.line_chart(forecast_df.set_index("Data"))
st.dataframe(forecast_df)

