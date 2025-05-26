from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# Cargar el CSV
df = pd.read_csv('features_con_onsets.csv')

# Separar características (X) y etiquetas (y)
X = df.drop('label', axis=1)  # Las columnas de características (MFCC, RMS, etc.)
y = df['label']  # La etiqueta (número de mentas)

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo base
rf = RandomForestClassifier(random_state=15, n_jobs=-1)

# Malla de hiperparámetros para búsqueda
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [20, 30, None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Búsqueda por Grid Search con validación cruzada
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',  # puedes cambiar a 'f1_macro' si hay desbalance
    n_jobs=-1,
    verbose=2
)

# Entrenar la búsqueda
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros encontrados
print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)

print("\nMejor precisión en validación cruzada:")
print(f"{grid_search.best_score_:.2f}")

# Usar el mejor modelo para hacer predicciones
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluar el modelo final
print("\nReporte de clasificación final:")
print(classification_report(y_test, y_pred))

accuracy = best_model.score(X_test, y_test)
print(f"Precisión final del modelo optimizado: {accuracy:.2f}")

joblib.dump(best_model, 'modelo_randomforest_mentas2.pkl')

###otras graficas

import matplotlib.pyplot as plt
import numpy as np

importancias = best_model.feature_importances_
nombres_caracteristicas = X.columns
indices = np.argsort(importancias)[::-1]

plt.figure(figsize=(12,6))
plt.title("Importancia de las características")
plt.bar(range(len(importancias)), importancias[indices], align="center")
plt.xticks(range(len(importancias)), nombres_caracteristicas[indices], rotation=90)
plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
disp.plot(cmap="Blues", xticks_rotation=90)
plt.title("Matriz de confusión")
plt.show()
