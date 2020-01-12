import tensorflow as tf
from functools import partial
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, MaxPool2D,
    Dropout, BatchNormalization, Activation,
    Reshape, Conv1D, GRU, Bidirectional
)
from tensorflow.keras.models import Model
import sys, os; 
sys.path.insert(0, os.path.abspath('..'));
from layers.residual import Residual
from loss_functions.ctc_loss import CTCLoss
from metrics.string_similarity import LevenshteinMetric

def CRNN(n_classes, batch_size, 
         optimizer_type="sgd", training=True,
         lr=.001, reg=1e-6, dropout_chance=0.2):
    
    DefaultConv2D = partial(Conv2D,
                            kernel_size=3,
                            kernel_initializer="he_normal",
                            kernel_regularizer=l2(reg),
                            padding="same")
    
    images = Input(shape=(128,32,1), name='images')
    x = Dropout(0.2)(images)
    
    x = DefaultConv2D(filters=64)(x) # (None, 128, 32, 64)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    x = DefaultConv2D(filters=64)(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    x = MaxPool2D(pool_size=2)(x) 
    
    for filters in [64] * 3:   # (None, 64, 16, 64)
        x = Residual(filters)(x)

    x = Residual(128, strides=2)(x) # (None, 32, 8, 128)
    for filters in [128] * 4:
        x = Residual(filters)(x)
        
    # CNN to RNN
    x = Reshape((32,1024))(x) # (None, 32, 2048)
    x = Conv1D(filters=64, 
               kernel_size=5,
               kernel_initializer="he_normal",
               padding="same")(x) # (None, 32, 64)
    
    # RNN
    x = Bidirectional(
            GRU(units=256, 
                return_sequences=True, 
                kernel_initializer="he_normal"))(x)  # (None, 32, 512)
    x = BatchNormalization()(x)
    
    x = Bidirectional(
            GRU(units=256, 
                return_sequences=True, 
                kernel_initializer="he_normal"))(x)  # (None, 32, 512)
    x = BatchNormalization()(x)
    
    y_pred = Dense(n_classes,
                   activation="softmax",
                   name="logits_layer")(x) # (None, 32, 63)
    
    model = Model(inputs=images, 
                  outputs=y_pred)
    
    logit_length = [model.layers[-1].output_shape[1]] * batch_size 
    
    # Optimizer
    if optimizer_type == "sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01)
    elif optimizer_type == "nesterov_sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01, nesterov=True)
    elif optimizer_type == "adam":
        optimizer = Adam(lr=lr)

    #TODO implement bleu score metric, implement ctc beam search loss
    model.compile(loss=CTCLoss(logit_length=logit_length),
                  optimizer=optimizer,
                  metrics=[LevenshteinMetric()])
    
    print(model.summary())
    
    return model

if __name__=="__main__":
    model = CRNN(63, 32)