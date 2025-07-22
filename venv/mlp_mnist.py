#import libraries
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#load the data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data
# print(y_train[0])
#y_train = y_train.astype('float32')
#y_test = y_test.astype('float32')

# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# Build the architecture

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Evaluate
model.evaluate(X_test, y_test)

# Predictions
sample_images = X_test[:5]
sample_labels = y_test[:5]
predictions = model.predict(sample_images)
result = np.argmax(predictions, axis=1)
print(result)

plt.subplots(1,5)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.title(f"Actual Label : {y_test[i]}\n Predicted Label : {result[i]}")
    plt.imshow(sample_images[i], cmap='gray')

plt.show()