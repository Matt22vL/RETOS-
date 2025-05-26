import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# === 1. Cargar datos
df = pd.read_csv('features_normalizadas.csv')

# === 2. Separar características y etiquetas
X = df.drop('label', axis=1)
y = df['label']

# === 3. Estandarizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === 5. Definir modelo base
rf = RandomForestRegressor(random_state=15, n_jobs=-1)

# === 6. GridSearch para regresión
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [20, None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'bootstrap': [True]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# === 7. Evaluación del modelo
y_pred = best_model.predict(X_test)

print("\n🛠️ Mejores parámetros encontrados:")
print(grid_search.best_params_)

print("\n📉 Métricas de regresión:")
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE (error absoluto medio): {mae:.2f}")
print(f"RMSE (raíz del error cuadrático medio): {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# === 8. Gráfico de predicción vs realidad
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.8)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Cantidad real de mentas")
plt.ylabel("Predicción del modelo")
plt.title("Comparación real vs predicción")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 9. Importancia de características
importancias = best_model.feature_importances_
nombres_caracteristicas = X.columns
indices = np.argsort(importancias)[::-1]

plt.figure(figsize=(12,6))
plt.title("Importancia de las características")
plt.bar(range(len(importancias)), importancias[indices])
plt.xticks(range(len(importancias)), nombres_caracteristicas[indices], rotation=90)
plt.tight_layout()
plt.show()

# === 10. Guardar modelo y scaler
joblib.dump(best_model, 'modelo_randomforest_regresion.pkl')
joblib.dump(scaler, 'scaler_regresion.pkl')
print("💾 Modelo y scaler de regresión guardados con éxito.")
