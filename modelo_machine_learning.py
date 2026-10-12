# pip install pandas scikit-learn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar los datos
iris = load_iris()
X = iris.data  # medidas de la flor
y = iris.target  # especie (0, 1, 2)

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Crear y entrenar el modelo
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# 4. Evaluar
y_pred = modelo.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))

# 5. Predecir una flor nueva
# Longitud cepal, ancho del sépalo, longitud del pétalo, ancho del pétalo
flor = [[0.1, 0.5, 0.4, 0.2]]
print("Especie:", iris.target_names[modelo.predict(flor)[0]])
