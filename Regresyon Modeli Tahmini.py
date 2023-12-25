# Gerekli kütüphaneleri içe aktarın
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Rastgele veri oluşturun (örnek olarak lineer bir ilişki)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Veriyi eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturun ve eğitin
model = LinearRegression()
model.fit(X_train, y_train)

# Eğitim setindeki performansı değerlendirin
y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
print(f"Eğitim Seti MSE: {mse_train}")

# Test setindeki performansı değerlendirin
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Test Seti MSE: {mse_test}")

# Modeli görselleştirin
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_test_pred, color='blue', linewidth=3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresyon Modeli Tahmini')
plt.show()
