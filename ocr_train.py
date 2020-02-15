import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
import wandb
from wandb.keras import WandbCallback
from data.data_loader import DataLoader
from models.crnn import CRNN

def get_callbacks(early_stopping_patience=None,
                  checkpoint_path=None,
                  reduce_lr_on_plateau_monitor=None,
                  reduce_lr_on_plateau_factor=None,
                  reduce_lr_on_plateau_patience=None,
                  reduce_lr_on_plateau_min_lr=None):  
    callbacks = []
    
    wandb_cb = WandbCallback()
    callbacks.append(wandb_cb)
    
    # Early Stopping
    if early_stopping_patience is not None:
        early_stopping = EarlyStopping(patience=early_stopping_patience,
                                       restore_best_weights=True)
        callbacks.append(early_stopping)
    
    # Model Checkpoints
    if checkpoint_path is not None:
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path, 
            save_weights_only=True,
            verbose=1, save_best_only=True)
        callbacks.append(checkpoint)
    
    #Reduce LR on Plateau
    if reduce_lr_on_plateau_monitor is not None:
        reduce_lr = ReduceLROnPlateau(#monitor='val_loss',
                                      factor=reduce_lr_on_plateau_factor,
                                      patience=reduce_lr_on_plateau_patience, 
                                      min_lr=reduce_lr_on_plateau_min_lr, 
                                      verbose=1)
        callbacks.append(reduce_lr)

    return callbacks

if __name__=="__main__":
    num_classes = 63 # 62 characters + 1 blank for ctc
    batch_size = 128
    seed = 23
    annotations_path="data/annotations.csv"
    images_path="data/word_images/data/"
    checkpoint_path = "checkpoints/crnn/" + \
        'epoch.{epoch:02d}_val_loss.{val_loss:.6f}.h5'
    data = DataLoader(batch_size,  
                      annotations_path=annotations_path, 
                      images_path=images_path,
                      seed=seed)
    
    train, steps_per_epoch = data.load_text_data()
    valid, validation_steps = data.load_text_data(type='valid')
    test, test_steps = data.load_text_data(type='test')
    
    cnn_weights_path = 'checkpoints/cnn_best_weights/' + \
        'epoch.152_val_loss.0.600990.h5'
# =============================================================================
#     
#     model = CRNN(num_classes,
#                  batch_size,
#                  lr=3e-4, 
#                  optimizer_type="adam",
#                  reg=1e-6,
#                  cnn_weights_path=cnn_weights_path)
#     
#     model.fit(train, 
#               epochs=100,
#               steps_per_epoch=steps_per_epoch,
#               validation_data=valid,
#               validation_steps=validation_steps,
#               verbose=1)
# =============================================================================
    
# =============================================================================
#     wandb.init(project="ocr",
#                name="crnn_overfit_7",
#                notes="Get the crnn to overfit to 1 word image ('untapering')",
#                config={
#                        "epochs": 100,
#                        "num_classes": num_classes,
#                        "batch_size": batch_size,
#                        "optimizer": "adam",
#                        "learning_rate": 3e-2,
#                        "l2_reg": 1e-3,
#                        "edit_dist_decoder": "greedy",
#                        })
#     config = wandb.config
#     
#     callbacks = get_callbacks()
#     
#     model = CRNN(n_classes=config.num_classes,
#                  batch_size=config.batch_size,
#                  lr=config.learning_rate, 
#                  optimizer_type=config.optimizer,
#                  reg=config.l2_reg,
#                  cnn_weights_path=cnn_weights_path)
#     
#     model.fit(train, 
#               epochs=config.epochs,
#               steps_per_epoch=steps_per_epoch,
#               validation_data=valid,
#               validation_steps=validation_steps,
#               callbacks=callbacks,
#               verbose=1)
# =============================================================================
    
    wandb.init(project="ocr",
               name="crnn_29k_train_1",
               notes="full crnn train",
               config={
                       "epochs": 500,
                       "num_classes": num_classes,
                       "batch_size": batch_size,
                       "optimizer": "adam",
                       "learning_rate": 3e-2,
                       "l2_reg": 1e-3,
                       "edit_dist_decoder": "greedy",
                       "early_stopping_patience": 20,
                       "reduce_lr_on_plateau_monitor": "val_loss",
                       "reduce_lr_on_plateau_min_lr": 1e-6,
                       "reduce_lr_on_plateau_factor":.33333,
                       "reduce_lr_on_plateau_patience": 10,
                       })
    config = wandb.config
    
    callbacks = get_callbacks(
        early_stopping_patience=config.early_stopping_patience,
        checkpoint_path=checkpoint_path,
        reduce_lr_on_plateau_monitor=config.reduce_lr_on_plateau_monitor,
        reduce_lr_on_plateau_factor=config.reduce_lr_on_plateau_factor,
        reduce_lr_on_plateau_patience=config.reduce_lr_on_plateau_patience,
        reduce_lr_on_plateau_min_lr=config.reduce_lr_on_plateau_min_lr
        )
    
    model = CRNN(n_classes=config.num_classes,
                 batch_size=config.batch_size,
                 lr=config.learning_rate, 
                 optimizer_type=config.optimizer,
                 reg=config.l2_reg,
                 cnn_weights_path=cnn_weights_path)
    
    model.fit(train, 
              epochs=config.epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid,
              validation_steps=validation_steps,
              callbacks=callbacks,
              verbose=1)
    

