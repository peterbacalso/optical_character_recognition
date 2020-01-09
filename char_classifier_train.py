import wandb
import numpy as np
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from data.data_loader import load_emnist
from models.one_hidden_layer import Simple_NN
from models.cnn import CNN

def get_callbacks(early_stopping_patience, 
                  reduce_lr_on_plateau_factor,
                  reduce_lr_on_plateau_patience,
                  reduce_lr_on_plateau_min_lr):   
# =============================================================================
#     # Tensorboard
#     root_logdir = os.path.join(os.curdir, 'logs')
#     run_id = time.strftime(f"run_%Y_%m_%d-%H_%M_%S")
#     run_logdir = os.path.join(root_logdir, run_id)
#     tensorboard = TensorBoard(run_logdir)
# =============================================================================
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
    batch_size = 128
    data_path="data/characters/emnist-byclass.mat"
    train, valid, test, num_classes, \
    steps_per_epoch, validation_steps \
    = load_emnist(batch_size, data_path=data_path)
    
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
                       "epochs": 200,
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
    
    model = CNN(num_classes, 
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
    
# =============================================================================
#     max_count = 100
#     for i in range(max_count):
#         
#         wandb.init(project="ocr",
#                    name=f"hyperparam_search_{i+1}",
#                    config={
#                            "epochs": 5,
#                            "optimizer": "sgd",
#                            "learning_rate": 10**np.random.uniform(-5,5),
#                            "l2_reg": 10**np.random.uniform(-3,-6)
#                            })
#         config = wandb.config
#         
#         model = Simple_NN(num_classes, 
#                           lr=config.learning_rate, 
#                           #optimizer=config.optimizer,
#                           reg=config.l2_reg)
#         model.fit(train, 
#                   epochs=config.epochs,
#                   steps_per_epoch=steps_per_epoch,
#                   validation_data=valid,
#                   validation_steps=validation_steps,
#                   callbacks=[WandbCallback()],
#                   verbose=2)
# =============================================================================
