import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score
import joblib

# === 1. Cargar datos enriquecidos
df = pd.read_csv('features_enriquecidas.csv')

# Verificar balance de clases
print("📊 Distribución de clases:")
print(df['label'].value_counts().sort_index())

# === 2. Separar características y etiquetas
X = df.drop('label', axis=1)
y = df['label']

# === 3. Estandarizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Dividir conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === 5. Modelo base + GridSearch
rf = RandomForestClassifier(random_state=15, n_jobs=-1)

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

print("\n📌 Mejores parámetros:")
print(grid_search.best_params_)

print("\n📈 Accuracy validación cruzada:")
print(f"{grid_search.best_score_:.3f}")

print("\n✅ Accuracy en test:")
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy:.3f}")

print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# === 7. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
disp.plot(cmap="Blues", xticks_rotation=90)
plt.title("Matriz de confusión")
plt.tight_layout()
plt.show()

# === 8. Importancia de características
importancias = best_model.feature_importances_
nombres_caracteristicas = X.columns
indices = np.argsort(importancias)[::-1]

plt.figure(figsize=(12,6))
plt.title("Importancia de las características")
plt.bar(range(len(importancias)), importancias[indices])
plt.xticks(range(len(importancias)), nombres_caracteristicas[indices], rotation=90)
plt.tight_layout()
plt.show()

# === 9. Precisión por clase
precision_por_clase = precision_score(y_test, y_pred, average=None, labels=sorted(y.unique()))
plt.figure(figsize=(12,4))
sns.barplot(x=sorted(y.unique()), y=precision_por_clase)
plt.title("Precisión por clase (número de mentas)")
plt.xlabel("Etiqueta real")
plt.ylabel("Precisión")
plt.show()

# === 10. Guardar modelo y scaler
joblib.dump(best_model, 'modelo_rf_enriquecido.pkl')
joblib.dump(scaler, 'scaler_enriquecido.pkl')
print("💾 Modelo y scaler guardados como 'modelo_rf_enriquecido.pkl' y 'scaler_enriquecido.pkl'")
