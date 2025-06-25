import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Preprocessing the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Building a simple neural network model
# This model is a basic feedforward neural network for image classification
model = Sequential([
    Flatten(input_shape=(32, 32, 3)), # 32 x 32 image size, 3 colour channels (RGB)
    Dense(1000, activation='relu'),  # Hidden layer with 1000 neurons
    Dense(10, activation='softmax')   # Output layer with 10 neurons (one for each class)
])

# Compiling and training the model
# Using Adam optimizer and categorical crossentropy loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))
model.save('cifar10_model.h5')