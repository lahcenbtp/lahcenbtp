-import pyopennn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Charger les données depuis le tableau
data = pd.DataFrame({
  "FNA": [14.5, 14.3, 14.2, 23.0, 23.0, 23.0],
  "SSF": [38.0, 38.0, 37.8, 31.5, 31.5, 31.0], 
  "SSA": [43.0, 42.8, 42.6, 40.5, 40.5, 40.0],
  "BC": [4.5, 5.0, 5.5, 4.5, 5.0, 5.5],
  "CoTh": [0.51, 0.74, 0.64, 0.72, 0.86, 0.79],
  "SpCa": [1.03, 1.30, 1.90, 1.27, 1.43, 1.67],
  "Fl": [2.45, 2.93, 2.78, 2.12, 2.70, 2.34],
  "St": [14.10, 23.49, 19.21, 11.89, 21.81, 18.02]  
})

# Diviser les données en ensembles d'entraînement et de test
X = data[["FNA", "SSF", "SSA", "BC", "CoTh", "SpCa"]]
y1 = data["Fl"]
y2 = data["St"]

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)

# Normaliser les données
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurer et entraîner le modèle pour la première sortie (Fl)
model_fl = nn.Sequential()
model_fl.add(nn.Dense(10, input_shape=(6,), activation='sigmoid'))
model_fl.add(nn.Dense(1))

model_fl.compile(loss='mean_squared_error', optimizer='adam')
model_fl.fit(X_train_scaled, y1_train, epochs=500, verbose=0)

# Prédire sur l'ensemble de test
y1_pred = model_fl.predict(X_test_scaled)

# Calculer et afficher les performances
mse_fl = mean_squared_error(y1_test, y1_pred)
print("Mean Squared Error for Fl:", mse_fl)

# Configurer et entraîner le modèle pour la deuxième sortie (St)
model_st = nn.Sequential()
model_st.add(nn.Dense(10, input_shape=(6,), activation='sigmoid'))
model_st.add(nn.Dense(1))

model_st.compile(loss='mean_squared_error', optimizer='adam')
model_st.fit(X_train_scaled, y2_train, epochs=500, verbose=0)

# Prédire sur l'ensemble de test
y2_pred = model_st.predict(X_test_scaled)

# Calculer et afficher les performances
mse_st = mean_squared_error(y2_test, y2_pred)
print("Mean Squared Error for St:", mse_st)

<!---
lahcenbtp/lahcenbtp is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
