import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import xgboost as xgb
import joblib

# === 1. Cargar datos enriquecidos
df = pd.read_csv('features_enriquecidas.csv')
X = df.drop('label', axis=1)
y = df['label']

# === 2. Codificar etiquetas de 1–29 a 0–28
le = LabelEncoder()
y_enc = le.fit_transform(y)

# === 3. Estandarizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# === 5. Definir XGBoost
xgb_model = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# === 6. GridSearchCV de hiperparámetros
param_grid = {
    'n_estimators':    [100, 200],
    'max_depth':       [3, 6],
    'learning_rate':   [0.01, 0.1],
    'subsample':       [0.8, 1.0],
    'colsample_bytree':[0.8, 1.0]
}

grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    error_score='raise'  # para que muestre errores al instante
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# === 7. Predicción y decodificación
y_pred_enc = best_model.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)     # vuelvo a 1–29
y_test_orig = le.inverse_transform(y_test)    # verdaderas 1–29

# === 8. Métricas
acc = accuracy_score(y_test, y_pred_enc)
print(f"\n🔑 Mejores parámetros: {grid.best_params_}")
print(f"📈 Accuracy CV: {grid.best_score_:.3f}")
print(f"✅ Accuracy test: {acc:.3f}\n")
print("📋 Reporte de clasificación:")
print(classification_report(y_test_orig, y_pred))

# === 9. Matriz de confusión
cm = confusion_matrix(y_test_orig, y_pred, labels=le.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=90)
plt.title("Matriz de confusión XGBoost")
plt.tight_layout()
plt.show()

# === 10. Importancia de características
importancias = best_model.feature_importances_
cols = X.columns
idxs = np.argsort(importancias)[::-1]

plt.figure(figsize=(12,6))
plt.title("Importancia de características (XGBoost)")
plt.bar(range(len(importancias)), importancias[idxs])
plt.xticks(range(len(importancias)), cols[idxs], rotation=90)
plt.tight_layout()
plt.show()

# === 11. Guardar modelo, scaler y label encoder
joblib.dump(best_model, 'modelo_xgboost_mentas_enriquecido.pkl')
joblib.dump(scaler,     'scaler_enriquecido_xgb.pkl')
joblib.dump(le,         'label_encoder_mentas.pkl')
print("🧠 Modelo, scaler y encoder guardados con éxito.")
