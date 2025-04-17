import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Carica dati
df = pd.read_csv("Dati_temperatura.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date")
data = df['Temperature'].values.reshape(-1, 1)

# Normalizza
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepara dataset
X, y = [], []
for i in range(30, len(data_scaled)):
    X.append(data_scaled[i-30:i])
    y.append(data_scaled[i])
X, y = np.array(X), np.array(y)

# Modello LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(30, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)

# Salva modello in formato .keras
model.save("modello_lstm.keras")

# Salva anche scaler
import joblib
joblib.dump(scaler, "scaler.save")