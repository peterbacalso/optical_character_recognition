import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
        Layer, Conv2D, BatchNormalization, LeakyReLU
)

class Residual(Layer):
    def __init__(self, filters, strides=1, reg=0.0, activation='relu', **kwargs):
        super().__init__(**kwargs)
        if activation == 'leaky_relu':
            self.activation = LeakyReLU(0.2)
        else:
            self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
                Conv2D(filters, kernel_size=3, kernel_regularizer=l2(reg),
                       strides=strides, padding="same", use_bias=False),
                BatchNormalization(),
                self.activation,
                Conv2D(filters, kernel_size=3, kernel_regularizer=l2(reg),
                       strides=1, padding="same", use_bias=False),
                BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                    Conv2D(filters, kernel_size=1, kernel_regularizer=l2(reg),
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
    def get_config(self):
        base_config = super().get_config()
        return base_config
