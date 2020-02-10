from functools import partial
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Flatten, MaxPool2D,
    Dropout, BatchNormalization, Activation,
    LeakyReLU
)
import sys, os; 
sys.path.insert(0, os.path.abspath('..'));
#from layers.residual import Residual

def LeNet(n_classes, optimizer_type="sgd",
        lr=.001, reg=1e-6, dropout_chance=0.2,
        channels=1, compile_model=True, 
        weights_path=None):
    
    DefaultConv2D = partial(Conv2D,
                            kernel_size=3,
                            kernel_initializer="he_normal",
                            kernel_regularizer=l2(reg),
                            padding="same")
    
    images = Input(shape=(32,32,1), name='images')
    
    x = DefaultConv2D(filters=6, kernel_size=5, 
                      padding='valid')(images) # (None, 28, 28, 6)
    #x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    #x = LeakyReLU(.01)(x)
    x = MaxPool2D(pool_size=2, strides=2)(x) # (None, 14, 14, 6)
    
    x = DefaultConv2D(filters=16, kernel_size=5, 
                      padding='valid')(x) # (None, 10, 10, 16)
    #x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    #x = LeakyReLU(.01)(x)
    x = MaxPool2D(pool_size=2, strides=2)(x) # (None, 5, 5, 16)
    
    x = Flatten()(x) # (None, 400)
    
    #x = Dropout(dropout_chance)(x)
    x = Dense(units=120, 
              kernel_regularizer=l2(reg))(x) # (None, 120)
    #x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    #x = LeakyReLU(.01)(x)
    
    #x = Dropout(dropout_chance)(x)
    x = Dense(units=84, 
              kernel_regularizer=l2(reg))(x) # (None, 120)
    #x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    #x = LeakyReLU(.01)(x)
    
    #x = Dropout(dropout_chance)(x)
    y_pred = Dense(n_classes, 
                   activation="softmax", 
                   name="classifier")(x) # (None, 62)
    
    model = Model(inputs=images, 
                  outputs=y_pred)
        
    # Optimizer
    if optimizer_type == "sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01)
    elif optimizer_type == "nesterov_sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01, nesterov=True)
    elif optimizer_type == "adam":
        optimizer = Adam(lr=lr)
        
    if compile_model:
        model.compile(loss="categorical_crossentropy", 
                      optimizer=optimizer, 
                      metrics=["accuracy"])
    
        print(model.summary())
    
    return model

if __name__=="__main__":
    model = LeNet(62)