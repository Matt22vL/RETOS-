import joblib

# Cargar el modelo ya entrenado
model = joblib.load('modelo_randomforest_mentas.pkl')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cargar el dataset solo para obtener los nombres de las columnas
df = pd.read_csv('features.csv')
X = df.drop('label', axis=1)

# Obtener las importancias
importancias = model.feature_importances_
nombres_caracteristicas = X.columns
indices = np.argsort(importancias)[::-1]

# Graficar
plt.figure(figsize=(12,6))
plt.title("Importancia de las características")
plt.bar(range(len(importancias)), importancias[indices], align="center")
plt.xticks(range(len(importancias)), nombres_caracteristicas[indices], rotation=90)
plt.tight_layout()
plt.show()
