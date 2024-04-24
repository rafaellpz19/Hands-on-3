import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


class RegresionPolinomial:
    def __init__(self, grado):
        self.grado = grado
        self.model = LinearRegression()
        self.poly = PolynomialFeatures(degree=grado)

    def entrenar(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)

    def predecir(self, X_pred):
        X_pred_poly = self.poly.transform(X_pred)
        return self.model.predict(X_pred_poly)

    def imprimir_ecuacion(self):
        coeficientes = self.model.coef_[1:]
        coeficiente_intercepto = self.model.intercept_

        ecuacion = f'y = {coeficiente_intercepto}'
        for i in range(len(coeficientes)):
            ecuacion += f' + {coeficientes[i]} * X^{i + 1}'

        print("Ecuación de Regresión Polinomial:")
        print(ecuacion)

    def calcular_r2(self, X, y):
        X_poly = self.poly.transform(X)
        y_pred = self.model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        return r2


# Datos de ejemplo
X = np.array([[118], [115], [106], [97], [95],[91], [97], [83], [83], [78],[54], [67], [56], [53], [61],[115], [81], [78], [30], [45],[99], [32], [25], [28], [90],[89]])
y = np.array([95, 96, 95, 97, 93, 94, 95,93, 92, 86,73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93])

# Crear objeto de regresión polinomial de grado 2
regresion = RegresionPolinomial(grado=2)

# Entrenar el modelo
regresion.entrenar(X, y)

# Imprimir ecuación de regresión
regresion.imprimir_ecuacion()

# Calcular y imprimir coeficiente de determinación (R2)
r2 = regresion.calcular_r2(X, y)
print("Coeficiente de determinación (R2) para regresión polinomial de grado 2:", r2)

# Predicciones
X_pred = np.array([[8],[12],[16]])
y_pred = regresion.predecir(X_pred)
print("Predicciones:")
for i in range(len(X_pred)):
    print(f'X = {X_pred[i][0]}, Y = {y_pred[i]}')