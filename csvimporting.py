import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load all CSV files
folder_path = "/Users/vichruth-victorious/Desktop/odbdata"
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Fix column names
df.columns = df.columns.str.strip().str.replace("()", "", regex=False)

# Convert problematic columns
#df["RELATIVE_THROTTLE_POSITION"] = pd.to_numeric(df["RELATIVE_THROTTLE_POSITION"], errors='coerce')

# Drop NaN values
df.dropna(inplace=True)

# Scale data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df.iloc[:, 1:].astype(np.float32))

print("Preprocessing Done! Shape:", df_scaled.shape)
print(df.columns)

# Define the Autoencoder
input_dim = df_scaled.shape[1]
encoding_dim = 8

# Encoder
input_layer = keras.Input(shape=(input_dim,))
encoded = layers.Dense(16, activation='relu')(input_layer)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(16, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

# Autoencoder Model
autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss='mse')

# Train the model
autoencoder.fit(df_scaled, df_scaled, epochs=50, batch_size=120, shuffle=True, validation_split=0.1)

# Detecting Anomalies
reconstructions = autoencoder.predict(df_scaled, verbose=0)
mse = np.mean(np.abs(df_scaled - reconstructions), axis=1)

# Set anomaly threshold
threshold = np.percentile(mse, 95)

# Detect anomalies
df["Anomaly"] = mse > threshold
print(df[df["Anomaly"] == True])

# Visualizing
plt.hist(mse, bins=50)
plt.axvline(threshold, color='red', linestyle='dashed')
plt.title("Anomaly Detection Threshold")
plt.show()
