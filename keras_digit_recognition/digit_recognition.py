import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])

model.fit(x_train, y_train, epochs=4)
model.save('digrec.keras')

model = tf.keras.models.load_model('digrec.keras')
# comment out lines 14-23 after running once

# outside data
# n = 1
# while os.path.isfile(f"digits/d{n}.png"):
#     img = cv2.imread(f"digits/d{n}.png")[:,:,0]
#     img = np.invert(np.array([img]))
#     prediction = model.predict(img)
#     print(f"This is probably a {np.argmax(prediction)}")
#     plt.imshow(img[0], cmap=plt.cm.binary)
#     plt.show()
#     n+=1

prediction = model.predict(x_test)

for _ in range(10):
    i = random.randint(0,9999)
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title(f"Prediction: {np.argmax(prediction[i])}")
    plt.xlabel(f"Actual: {y_test[i]}")
    plt.show()