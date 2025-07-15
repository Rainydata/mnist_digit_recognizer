import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load.data()

#normalize pixels values (0 to 1)

X_train, X_test = X_train / 255.0, X_test / 255.0

#define the model

