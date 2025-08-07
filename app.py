from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import base64
import cv2
from io import BytesIO
from PIL import Image
import re

app = Flask(__name__)

model = load_model("model/modelo_mnist.h5")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Obtener imagen base64 y decodificar
        image_data = data["image"]
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        img_bytes = base64.b64decode(image_data)
        

        # Convertir a imagen en escala de grises
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        # Redimensionar a 28x28
        img = cv2.resize(img, (28, 28))

        # Invertir colores (fondo negro, número blanco)
        img = cv2.bitwise_not(img)

        # Normalizar a [0,1]
        img = img.astype("float32") / 255.0

        # Asegurar que tenga forma (1,28,28,1)
        img = img.reshape(1, 28, 28, 1)
        cv2.imwrite("debug_img.png", img * 255)  # temporal para revisar

        # Hacer predicción
        prediction = model.predict(img)
        digit = int(np.argmax(prediction))

        return jsonify({"digit": digit})

    except Exception as e:
        return jsonify({"Error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)