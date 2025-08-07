import pandas as pd
import seaborn as sns
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from collections import Counter
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

data_path = "../dataset/train.csv"

data = pd.read_csv(data_path)
print(data.head())
data.head()
print(data.isnull().sum())
#data.describe()

#view the image

pixels = data.iloc[0, 1:].values.reshape(28,28)
plt.imshow(pixels, cmap="gray")
plt.title(f"label: {data.iloc[0, 0]}")
plt.axis("off")
#plt.show()


##clean null values

print("valores nulos por columna:\n", data.isnull().sum())

## verify the clases distribution (balance de digitos) 

print("conteo de balanceo de las clase:\n", data['label'].value_counts())

## class weighting o ponderacion de clases, es para ajustar los pesos de la clase 5 ya que tiene menos datos que los demas
##“Oye, si fallas el dígito 5, te va a doler más que si fallas un 1”.

class_count = Counter(data["label"])
print(class_count)

## all clases

classes = np.unique(data["label"])
print(classes)

##calculate the inverse weight

weights = compute_class_weight(class_weight="balanced", classes=classes, y = data["label"])

#convert two lists (classes and weights to dict
class_weight = dict(zip(classes, weights))

print(class_weight)

#convert pixels value (0-255) to small values for trainnig faster and eficcient
X = data.drop("label", axis=1).values / 255.0#normalizacion
y = data["label"].values#label

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#convert images to model

X_train = X_train.reshape(-1,28,28,1)
X_val = X_val.reshape(-1,28,28,1)