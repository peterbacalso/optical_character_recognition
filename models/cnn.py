from functools import partial
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
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
    
    DefaultConv2D = partial(Conv2D,
                            kernel_size=3,
                            kernel_initializer="he_normal",
                            #kernel_regularizer=l2(reg),
                            padding="same")
    
    images = Input(shape=(32,16,1), name='images')
    #x = Dropout(0.2)(images)
    
    x = DefaultConv2D(kernel_size=5,filters=64)(images) # (None, 32, 16, 64)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    x = DefaultConv2D(filters=64)(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    x = MaxPool2D(pool_size=(2,1))(x) # (None, 16, 16, 64)
    
    for filters in [64] * 4:
        x = Residual(filters)(x)

    x = Residual(128, strides=2)(x) # (None, 8, 8, 128)
    for filters in [128] * 5:
        x = Residual(filters)(x)
        
    x = Residual(256, strides=2)(x) # (None, 4, 4, 256)
    for filters in [256] * 5:
        x = Residual(filters)(x)
        
    x = GlobalAvgPool2D()(x)
    
    y_pred = Dense(n_classes, 
                   activation="softmax", 
                   name="classifier")(x)
    
    model = Model(inputs=images, 
                  outputs=y_pred)
        
        # Optimizer
    if optimizer_type == "sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01)
    elif optimizer_type == "nesterov_sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01, nesterov=True)
    elif optimizer_type == "adam":
        optimizer = Adam(lr=lr)
        
    model.compile(loss="categorical_crossentropy", 
                  optimizer=optimizer, 
                  metrics=["accuracy"])
    
    print(model.summary())
    
    return model

if __name__=="__main__":
    model = CNN(62)