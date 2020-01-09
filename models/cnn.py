from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Flatten, MaxPool2D,
    Dropout, BatchNormalization, Activation, GlobalAvgPool2D
)
import sys, os; 
sys.path.insert(0, os.path.abspath('..'));
from layers.residual import Residual

def CNN(n_classes, optimizer_type="sgd",
        lr=.001, reg=1e-6, dropout_chance=0.2,
        channels=1):
    
    # Optimizer
    if optimizer_type == "sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01)
    elif optimizer_type == "nesterov_sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01, nesterov=True)
    elif optimizer_type == "adam":
        optimizer = Adam(lr=lr)

    model = Sequential([
        Input(shape=(28,28,channels)),
        Dropout(0.2),
        
        Conv2D(filters=64,
               kernel_size=3,
               kernel_initializer="he_normal",
               kernel_regularizer=l2(reg),
               padding="same"),
        BatchNormalization(),
        Activation(activation="relu"),
        Conv2D(filters=64,
               kernel_size=3,
               kernel_initializer="he_normal",
               kernel_regularizer=l2(reg),
               padding="same"),
        BatchNormalization(),
        Activation(activation="relu"),
        MaxPool2D(pool_size=2),
        
        Residual(filters=64),
        Residual(filters=64),
        Residual(filters=64),
        Residual(filters=128, strides=2),
        Residual(filters=128),
        Residual(filters=128),
        Residual(filters=128),
        GlobalAvgPool2D(),
        #Flatten(),
        Dropout(dropout_chance),
        Dense(n_classes, 
              activation="softmax", 
              name="classifier")
    ])
    model.compile(loss="categorical_crossentropy", 
                  optimizer=optimizer, 
                  metrics=["accuracy"])
    
    print(model.summary())
    
    return model
