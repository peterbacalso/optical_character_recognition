import tensorflow as tf
from functools import partial
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, MaxPool2D,
    Dropout, BatchNormalization, Activation,
    Reshape, Conv1D, GRU, Bidirectional,
    Lambda, Conv2DTranspose, TimeDistributed
)
from tensorflow.keras.models import Model
import sys, os; 
sys.path.insert(0, os.path.abspath('..'));
from layers.identity_conv import identity_conv
from layers.padder import padder
from layers.expander import expander
from loss_functions.ctc_loss import CTCLoss
from metrics.string_similarity import LevenshteinMetric
from models.cnn import CNN

def CRNN(n_classes, batch_size, 
         optimizer_type="sgd", training=True,
         lr=.001, reg=1e-6, dropout_chance=0.2,
         cnn_weights_path=None):
    
    images = Input(shape=(32,128,1), name='images')
    #x = Dropout(0.2)(images)
    
    # extract patches of 32hx16w with stride 2
    x = Lambda(identity_conv, 
               arguments={'patch_size': (32,16), 'stride': 2})(images)
    
    num_windows = x.shape[1]
    x = Reshape((num_windows,32,16,1))(x)
    
    # expand width-wise 32x16 to 32x32
    x = TimeDistributed(Lambda(expander), name="expander")(x) # (None, 32, 32, 1)
    
# =============================================================================
#     # pad width-wise with 0's to expand 32x16 to 32x32
#     x = TimeDistributed(Lambda(padder), name="padder")(x) # (None, 32, 32, 1)
# =============================================================================
    
# =============================================================================
#     # expand 32x16 to 32x32 deconvolution
#     x = TimeDistributed(Conv2DTranspose(filters=1, 
#                         kernel_size=3,
#                         strides=(1,2), 
#                         padding='same'), name='expand_patch')(x) # (None, 32, 32, 1)
# =============================================================================
    
    cnn = CNN(n_classes-1, reg=reg, compile_model=False)
    if cnn_weights_path:
        cnn.load_weights(cnn_weights_path)
    cnn = Model(inputs=cnn.inputs, outputs=cnn.layers[-2].output)
    #cnn.trainable = False
    x = TimeDistributed(cnn, name='convnet')(x) # (None, num_windows, 512) 

# =============================================================================
#     # CNN to RNN
#     x = Conv1D(filters=256, 
#                kernel_size=3,
#                kernel_initializer="he_normal",
#                padding="same")(x) # (None, num_windows, 256)
# =============================================================================
    
    # RNN
    x = Bidirectional(
            GRU(units=256, 
                return_sequences=True, 
                kernel_initializer="he_normal"))(x)  # (None, num_windows, 512)
    x = BatchNormalization()(x)
    
# =============================================================================
#     x = Bidirectional(
#             GRU(units=256, 
#                 return_sequences=True, 
#                 kernel_initializer="he_normal"))(x)  # (None, num_windows, 512)
#     x = BatchNormalization()(x)
# =============================================================================
    
    
    y_pred = Dense(n_classes,
                   activation="softmax",
                   name="logits_layer")(x) # (None, num_windows, 63)
    
    model = Model(inputs=images, 
                  outputs=y_pred)
    
    logit_length = [[model.layers[-1].output_shape[1]] * batch_size]
    
    # Optimizer
    if optimizer_type == "sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01)
    elif optimizer_type == "nesterov_sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01, nesterov=True)
    elif optimizer_type == "adam":
        optimizer = Adam(lr=lr)

    model.compile(loss=CTCLoss(logit_length=logit_length),
                  optimizer=optimizer,
                  metrics=[LevenshteinMetric(batch_size=batch_size)])
    
    print(model.summary())
    
    return model

if __name__=="__main__":
    model = CRNN(63, 32)