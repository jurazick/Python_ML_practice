import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

data = keras.datasets.fashion_mnist

(train_imgs, train_labels), (test_imgs, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_imgs, train_labels, epochs=50)
model.save('clothing.keras')

model = tf.keras.models.load_model('clothing.keras')
# comment out lines 17-26 after running once

prediction = model.predict(test_imgs)

for _ in range(10):
    i = random.randint(0,9999)
    plt.grid(False)
    plt.imshow(test_imgs[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()