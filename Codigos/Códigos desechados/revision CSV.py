import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el CSV
df = pd.read_csv("features1.csv")

# Mostrar las primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Ver información general
print("\nInformación general:")
print(df.info())

# Estadísticas descriptivas
print("\nEstadísticas:")
print(df.describe())

# Ver si hay valores nulos
print("\n¿Hay valores faltantes?")
print(df.isnull().sum())

# Ver distribución de clases (cantidad de mentas)
plt.figure(figsize=(10, 4))
sns.countplot(x='label', data=df)
plt.title('Distribución de grabaciones por cantidad de mentas')
plt.xlabel('Cantidad de mentas')
plt.ylabel('Número de audios')
plt.tight_layout()
plt.show()
