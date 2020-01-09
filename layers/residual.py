import tensorflow as tf
from tensorflow.keras.layers import (
        Layer, Conv2D, BatchNormalization, LeakyReLU
)

class Residual(Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        if activation == 'leaky_relu':
            self.activation = LeakyReLU(0.2)
        else:
            self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
                Conv2D(filters, kernel_size=3, 
                       strides=strides, padding="same", use_bias=False),
                BatchNormalization(),
                self.activation,
                Conv2D(filters, kernel_size=3, 
                       strides=1, padding="same", use_bias=False),
                BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                    Conv2D(filters, kernel_size=1, 
                           strides=strides, padding="same", use_bias=False),
                    BatchNormalization()
            ]
    
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
