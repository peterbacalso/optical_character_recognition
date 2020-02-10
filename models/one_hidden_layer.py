import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Dense, Flatten
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD

def Simple_NN(n_classes, lr=.01, reg=0.0):
    weight_initer = tf.compat.v1.truncated_normal_initializer(mean=0.0, 
                                                              stddev=0.0001)
    W = tf.compat.v1.get_variable(name="Weight", 
                                  dtype=tf.float32, shape=[1024, 32], 
                                  initializer=weight_initer)
    W = tf.constant_initializer(W.numpy())
    optimizer = SGD(lr=lr)

    model = Sequential([
        Input(shape=(32,32,1)),
        Flatten(),
        Dense(units=32, 
              kernel_initializer=W, 
              bias_initializer="zeros",
              kernel_regularizer=l2(reg),
              activation="relu"),
        Dense(n_classes, 
              activation="softmax", 
              name="classifier")
    ])
    model.compile(loss="categorical_crossentropy", 
                  optimizer=optimizer, 
                  metrics=["accuracy"])
    
    print(model.summary())
    
    return model