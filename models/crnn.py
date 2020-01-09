import tensorflow as tf
from functools import partial
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, MaxPool2D,
    Dropout, BatchNormalization, Activation,
    Reshape, Conv1D, GRU, Bidirectional,
    Lambda
)
from tensorflow.keras.models import Model
import sys, os; 
sys.path.insert(0, os.path.abspath('..'));
from layers.residual import Residual
from loss_functions.ctc_loss import ctc_loss

def CRNN(n_classes, optimizer_type="sgd", training=True,
         lr=.001, reg=1e-6, dropout_chance=0.2):
    
    DefaultConv2D = partial(Conv2D,
                            kernel_size=3,
                            kernel_initializer="he_normal",
                            kernel_regularizer=l2(reg),
                            padding="same")
    
    labels = Input((31,), name='labels', dtype='int64') # (None , 31) Max text length is 31
    input_length = Input((), name='input_length', dtype='int32') # (None)
    label_length = Input((), name='label_length', dtype='int32') # (None)

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
              activation="softmax")(x) # (None, 32, 63)
    
    loss_out = Lambda(ctc_loss, 
                      output_shape=(1,), 
                      name='ctc')([labels, y_pred, input_length, label_length]) #(None, 1)
    
    if training:
        model = Model(inputs=[images, labels, input_length, label_length], 
                      outputs=loss_out)
    else:
        model = Model(inputs=[images], outputs=y_pred)
    
# =============================================================================
#     ctc_loss_fn = partial(ctc_loss,
#                           labels=labels,
#                           label_length=label_length,
#                           input_length=input_length)
# =============================================================================
    
    # Optimizer
    if optimizer_type == "sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01)
    elif optimizer_type == "nesterov_sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01, nesterov=True)
    elif optimizer_type == "adam":
        optimizer = Adam(lr=lr)

# =============================================================================
#     #TODO implement bleu score metric, implement ctc beam search loss
#     model.compile(loss=ctc_loss_fn,
#                   optimizer=optimizer)
# =============================================================================
        
    #TODO implement bleu score metric, implement ctc beam search loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                  optimizer=optimizer)
    
    print(model.summary())
    
    return model

if __name__=="__main__":
    CRNN(63)