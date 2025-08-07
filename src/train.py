from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import preprocessing

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 clases
])

model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    preprocessing.X_train, preprocessing.y_train,
    validation_data=(preprocessing.X_val, preprocessing.y_val),
    epochs=10,
    batch_size=64,
    class_weight=preprocessing.class_weight  # pesos para balancear clases
)

# 4. Evaluar el modelo
val_loss, val_acc = model.evaluate(preprocessing.X_val, preprocessing.y_val)
print(f"\n✅ Accuracy en validación: {val_acc:.4f}")

# 5. Guardar el modelo entrenado
model.save("../model/modelo_mnist.h5")
print("📦 Modelo guardado como 'modelo_mnist.h5'")
