import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Cargar el modelo previamente entrenado
model = load_model("../model/modelo_mnist.h5")

# --- Cargar una imagen nueva (ejemplo: una imagen de 28x28 píxeles en blanco y negro) ---
# Cambia esta ruta por la imagen que quieras probar
image_path = "cinco.png"

# Leer imagen en escala de grises y redimensionar a 28x28
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"No se encontró la imagen en: {image_path}")

img = cv2.resize(img, (28, 28))

# Invertir colores si es fondo blanco y número negro
img = 255 - img

# Normalizar y dar forma
img = img / 255.0
img = img.reshape(1, 28, 28, 1)

# Predecir
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

# Mostrar resultado
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f"Predicción: {predicted_class}")
plt.axis("off")
plt.show()
