from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Import data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the architecture

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics = ["accuracy"])

# Train 

model.fit(X_train, y_train, epochs=10, batch_size=64)

#  Evaluate
model.evaluate(X_test, y_test)