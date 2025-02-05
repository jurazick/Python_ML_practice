import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
print(len(test_data))
word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(val, key) for(key, val) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])

model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
model.save("imdbmodel.keras")

model = tf.keras.models.load_model('imdbmodel.keras')
# comment out lines 26-43 after running once

predictions = model.predict(test_data)

def classify_review(score):
    if score >= .5:
        return "(Positive)"
    return "(Negative)"

while True:
    rand_review = random.randint(0,24999)
    print("Review:", decode_review(test_data[rand_review]))
    print("Prediction:", predictions[rand_review][0], classify_review(predictions[rand_review][0]))
    print("Actual:", test_labels[rand_review], classify_review(test_labels[rand_review]))
    if input("Space to continue, q to quit").lower() == "q":
        break
    