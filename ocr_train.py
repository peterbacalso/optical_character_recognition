import numpy as np
from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
import wandb
from wandb.keras import WandbCallback
from data.data_loader import DataLoader
from models.crnn import CRNN

def get_callbacks(early_stopping_patience, 
                  reduce_lr_on_plateau_factor,
                  reduce_lr_on_plateau_patience,
                  reduce_lr_on_plateau_min_lr):   

    wandb_cb = WandbCallback()
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, 
                                   restore_best_weights=True)
    
     # Model Checkpoints
    checkpoint = ModelCheckpoint(
        filepath=f'checkpoints/' + \
        'epoch.{epoch:02d}_val_loss.{val_loss:.6f}.h5', 
        verbose=1, save_best_only=True)
    
    #Reduce LR on Plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=reduce_lr_on_plateau_factor,
                                  patience=reduce_lr_on_plateau_patience, 
                                  min_lr=reduce_lr_on_plateau_min_lr, 
                                  verbose=1)
    
    return [wandb_cb, early_stopping, checkpoint, reduce_lr]

if __name__=="__main__":
    num_classes = 63 # 62 characters + 1 blank for ctc
    batch_size = 128
    data_path_1="data/annotations.csv"
    data_path_2="data/y_true.csv"
    data_path_3="data/word_images/data/"
    data = DataLoader(batch_size, 
                      data_path_1=data_path_1, 
                      data_path_2=data_path_2,
                      data_path_3=data_path_3)
    
    train, steps_per_epoch = data.load_text_data()
    valid, validation_steps = data.load_text_data(type='valid')
    
# =============================================================================
#     model = CNN(num_classes, 
#                 lr=1e-2, 
#                 optimizer_type="sgd",
#                 reg=1e-6)
#     model.fit(train, 
#               epochs=5,
#               steps_per_epoch=steps_per_epoch,
#               validation_data=valid,
#               validation_steps=validation_steps,
#               verbose=1)
# =============================================================================
    
    wandb.init(project="ocr",
               config={
                       "epochs": 1,
                       "optimizer": "adam",
                       "learning_rate": 3e-4,
                       "l2_reg": 1e-3,
# =============================================================================
#                        "early_stopping_patience": 10,
#                        "reduce_lr_on_plateau_min_lr": 1e-6,
#                        "reduce_lr_on_plateau_factor":.33333,
#                        "reduce_lr_on_plateau_patience": 5,
# =============================================================================
                       })
    config = wandb.config
    
# =============================================================================
#     callbacks = get_callbacks(
#         early_stopping_patience=config.early_stopping_patience,
#         reduce_lr_on_plateau_factor=config.reduce_lr_on_plateau_factor,
#         reduce_lr_on_plateau_patience=config.reduce_lr_on_plateau_patience,
#         reduce_lr_on_plateau_min_lr=config.reduce_lr_on_plateau_min_lr
#         )
# =============================================================================

    callbacks=[WandbCallback()]
    
    model = CRNN(num_classes,
                 lr=config.learning_rate, 
                 optimizer_type=config.optimizer,
                 reg=config.l2_reg)
    
    model.fit(train, 
              epochs=config.epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid,
              validation_steps=validation_steps,
              callbacks=callbacks,
              verbose=1)
    
    