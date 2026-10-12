# pip install pandas numpy scikit-learn matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------- 1. Crear dataset de muestra ----------
# Fijamos una semilla aleatoria para asegurarnos de que los datos generados sean idénticos siempre
np.random.seed(42)
n = 150  # Cantidad de estaciones de transporte que vamos a simular

# Creamos un DataFrame con datos simulados (pasajeros, tiempos de espera, retrasos, etc.)
df = pd.DataFrame(
    {
        "estacion": [f"Estacion_{i+1}" for i in range(n)],
        "pasajeros_dia": np.random.randint(500, 25000, n),
        "pasajeros_hora_pico": np.random.randint(100, 5000, n),
        "tiempo_espera_min": np.round(np.random.uniform(1, 15, n), 1),
        "retraso_promedio_min": np.round(np.random.uniform(0, 10, n), 1),
        "buses_por_hora": np.random.randint(5, 40, n),
    }
)
# Guardamos los datos en un archivo CSV y mostramos las primeras 5 filas por consola
df.to_csv("dataset_transporte.csv", index=False)
print("--- Primeras filas del Dataset ---")
print(df.head())

# ---------- 2. Preparar los datos ----------
# Seleccionamos únicamente las columnas numéricas que servirán de variables para agrupar
variables = [
    "pasajeros_dia",
    "pasajeros_hora_pico",
    "tiempo_espera_min",
    "retraso_promedio_min",
    "buses_por_hora",
]

# Estandarizamos las variables (media = 0, desviación estándar = 1)
# Esto es vital para que K-Means no le dé más importancia a columnas con números grandes (como pasajeros)
X = StandardScaler().fit_transform(df[variables])

# ---------- 3. Elegir el número de grupos (k) ----------
print("\n--- Evaluación del puntaje de Silueta para cada k ---")
# Probamos agrupando los datos desde 2 hasta 7 clústeres diferentes
for k in range(2, 8):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    # Imprimimos la métrica de silueta (valores más cercanos a 1 indican mejor separación entre grupos)
    print(f"k={k}  silhouette={silhouette_score(X, modelo.labels_):.3f}")

# ---------- 4. Entrenar el modelo final ----------
k = 3  # Seleccionamos k=3 tras evaluar los resultados de la silueta
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

# Entrenamos el modelo y asignamos a cada estación su número de grupo correspondiente (0, 1 o 2)
df["grupo"] = kmeans.fit_predict(X)

# ---------- 5. Interpretar los grupos ----------
print("\n--- Promedios operativos por cada Grupo ---")
# Calculamos la media de las variables originales para cada grupo y redondeamos a 1 decimal
print(df.groupby("grupo")[variables].mean().round(1))

print("\n--- Conteo de estaciones por Grupo ---")
# Vemos cuántas estaciones cayeron dentro de cada grupo
print(df["grupo"].value_counts())

# ---------- 6. Gráfica (2D con PCA) ----------
# Como tenemos 5 variables (5 dimensiones), usamos PCA para reducir los datos a solo 2 dimensiones (componentes)
coords = PCA(n_components=2).fit_transform(X)

# Generamos un gráfico de dispersión (puntos) donde el color representa el grupo asignado a cada estación
plt.scatter(coords[:, 0], coords[:, 1], c=df["grupo"], cmap="viridis")
plt.title("Grupos de estaciones (K-Means)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()
