# Import the necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define a custom layer for an equivariant hidden layer
class EquivariantHiddenLayer(keras.layers.Layer):
    
    # Initialize the layer
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    # Build the layer with input shape
    def build(self, input_shape):
        # Initialize the layer's weights with random normal distribution with mean 0 and standard deviation 0.2, and seed 42
        initializer = tf.keras.initializers.RandomNormal(seed=42, stddev=0.2)
        # Define the layer's weights
        self.a = self.add_weight(
            name="w1", shape=(), initializer=initializer, trainable=True
        )
        self.b = self.add_weight(
            name="w2", shape=(), initializer=initializer, trainable=True
        )
        self.c = self.add_weight(
            name="w3", shape=(), initializer=initializer, trainable=True
        )
    
    # Define layer call with input as input tensor
    def call(self, inputs):
        # Define the weight matrix W1 with the learned weights
        W1 = [
            [self.a, self.b, self.c, self.b],
            [self.b, self.a, self.b, self.c],
            [self.c, self.b, self.a, self.b],
            [self.b, self.c, self.b, self.a],
        ]
        # Multiply input tensor with weight matrix
        return tf.matmul(inputs, W1)

# Define a custom layer for an equivariant output layer
class EquivariantOutputLayer(keras.layers.Layer):
    
    # Initialize the layer with trainable weight d
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define the layer's weight with a constant value of 1
        self.d = self.add_weight(name="d", shape=(), initializer="ones", trainable=True)
    
    # Define layer call with input as input tensor
    def call(self, inputs):
        # Calculate the first term by scaling the input with [1,-1,-1,1], and summation over rows
        first_term = self.d * tf.reduce_sum(
            inputs * [1, -1, -1, 1], axis=1, keepdims=True
        )
        # Calculate the second term by element-wise squaring and summation over rows
        second_term = tf.reduce_sum(inputs**2, axis=1, keepdims=True)
        # Concatenate the first and second term along axis=1 to get the output tensor
        return tf.concat([first_term, second_term], axis=1)
    
    
def eq_build_model(hp):
    model = tf.keras.Sequential()
    model.add(EquivariantHiddenLayer())
    model.add(EquivariantHiddenLayer())
    model.add(EquivariantOutputLayer())
    # Set batch size, number of epochs, optimizer, and learning rate
    batch_size = hp.Int("batch_size", min_value=16, max_value=256, step=16)
    num_epochs = hp.Int("num_epochs", min_value=10, max_value=200, step=10)
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=["accuracy", "mae"],
    )

    return model
