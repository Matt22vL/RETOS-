import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib

# === 1. Cargar nuevo dataset robusto
df = pd.read_csv('features_robustas_TOTAL.csv')

print("Distribución de clases:")
print(df['label'].value_counts().sort_index())

# === 2. Separar características y etiquetas
X = df.drop('label', axis=1)
y = df['label']

# === 3. Estandarizar nuevas características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. División train/test42
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=y)

# === 5. Modelo base y búsqueda de hiperparámetros
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

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
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# === 6. Evaluación
y_pred = best_model.predict(X_test)

print("\n📊 Mejores parámetros encontrados:")
print(grid_search.best_params_)

print("\n📈 Precisión validación cruzada:")
print(f"{grid_search.best_score_:.3f}")

print("\n✅ Precisión en test:")
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy:.3f}")

print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# === 7. Guardar modelo y scaler actualizados
joblib.dump(best_model, 'modelo_randomforest_robusto1.pkl')
joblib.dump(scaler, 'scaler_robusto1.pkl')
print("🧠 Modelo y scaler robustos guardados con éxito.")
