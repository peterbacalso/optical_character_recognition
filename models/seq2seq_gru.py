import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional, Conv1D, Conv2D, GRU, TimeDistributed, Dense
)
from tensorflow.keras.optimizers import SGD, Adam


def Seq2SeqGRU(n_classes, optimizer_type="sgd", lr=.001):

    # Optimizer
    if optimizer_type == "sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01)
    elif optimizer_type == "nesterov_sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01, nesterov=True)
    elif optimizer_type == "adam":
        optimizer = Adam(lr=lr)

    model = Sequential([
            Conv1D(filters=20, kernel_size=4, strides=2, 
                   padding="valid", input_shape=[None, 1]),
            Bidirectional(GRU(filters=20, return_sequences=True)),
            Bidirectional(GRU(filters=20, return_sequences=True)),
            TimeDistributed(Dense(10))
            
    ])
    model.compile(loss="mse", 
                  optimizer=optimizer, 
                  metrics=["accuracy"])
    
    print(model.summary())
    
    return model
