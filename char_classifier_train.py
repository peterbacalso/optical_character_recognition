#import wandb
import numpy as np
import time
#from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from data.char_loader import CharLoader
from models.one_hidden_layer import Simple_NN
from models.cnn import CNN
from models.lenet import LeNet

# =============================================================================
# def get_callbacks(early_stopping_patience=None,
#                   checkpoint_path=None,
#                   reduce_lr_on_plateau_monitor=None,
#                   reduce_lr_on_plateau_factor=None,
#                   reduce_lr_on_plateau_patience=None,
#                   reduce_lr_on_plateau_min_lr=None):  
#     callbacks = []
#     
#     wandb_cb = WandbCallback()
#     callbacks.append(wandb_cb)
#     
#     # Early Stopping
#     if early_stopping_patience is not None:
#         early_stopping = EarlyStopping(patience=early_stopping_patience,
#                                        restore_best_weights=True)
#         callbacks.append(early_stopping)
#     
#     # Model Checkpoints
#     if checkpoint_path is not None:
#         checkpoint = ModelCheckpoint(
#             filepath=checkpoint_path, 
#             save_weights_only=True,
#             verbose=1, save_best_only=True)
#         callbacks.append(checkpoint)
#     
#     #Reduce LR on Plateau
#     if reduce_lr_on_plateau_monitor is not None:
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss',
#                                       factor=reduce_lr_on_plateau_factor,
#                                       patience=reduce_lr_on_plateau_patience, 
#                                       min_lr=reduce_lr_on_plateau_min_lr, 
#                                       verbose=1)
#         callbacks.append(reduce_lr)
# 
#     return callbacks
# =============================================================================

if __name__=="__main__":
    batch_size = 128
    emnist_path = "data/characters/emnist-byclass.mat"
    fonts_path = "data/char_font_dataset.csv"
    checkpoint_path = f'checkpoints/cnn/' + \
        'epoch.{epoch:02d}_val_loss.{val_loss:.6f}.h5'
    
    data = CharLoader(batch_size,
                      dataset_size=20000,
                      emnist_path=emnist_path,
                      fonts_path=fonts_path)
    
    train, steps_per_epoch, num_classes = data.load_data(augment=True)
    valid, validation_steps, _ = data.load_data(type='valid')
    test, test_steps, _ = data.load_data(type='test')
    
     
# =============================================================================
#     model = CNN(num_classes, 
#                 compile_model=False)
#     model.load_weights('checkpoints/cnn_best_weights/' + \
#         'epoch.152_val_loss.0.600990.h5')
#     model.compile(loss="categorical_crossentropy", 
#                   optimizer='adam', 
#                   metrics=["accuracy"])
#     loss,acc = model.evaluate_generator(test, steps=test_steps, verbose=2)
#     print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# =============================================================================
    
    model = LeNet(num_classes, 
                  lr=3e-4, 
                  optimizer_type="adam",
                  reg=1e-3)
    #model = Simple_NN(num_classes, lr=.01, reg=0.0)
    
    start = time.time()
    model.fit(train, 
              epochs=10,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid,
              validation_steps=validation_steps,
              verbose=1)
    end = time.time()
    print(end - start)
    
# =============================================================================
#     # Experiments
#     experiment_notes = "Experiment 7: Modified LeNet
#     
# # =============================================================================
# #     experiment_2_notes = "Experiment 1: (Additional Dense layer)\n" + 
# #     "Dropout | Conv-Batch-LeakyReLU-MaxPool x 5 |" + 
# #     "Flatten | Dropout | Dense x 2 | Softmax"
# # =============================================================================
#     
#     wandb.init(project="ocr",
#                name="lenet_experiment_1",
#                notes=experiment_notes,
#                config={
#                        "epochs": 10,
#                        "optimizer": "adam",
#                        "learning_rate": 3e-4,
#                        "l2_reg": 1e-2,
#                        "dropout_chance": 0.5,
#                        })
#     config = wandb.config
#     
#     callbacks = get_callbacks()
#     
# # =============================================================================
# #     model = CNN(num_classes, 
# #                 lr=config.learning_rate, 
# #                 dropout_chance=config.dropout_chance,
# #                 optimizer_type=config.optimizer,
# #                 reg=config.l2_reg)
# # =============================================================================
#     
#     model = LeNet(num_classes, 
#                   lr=config.learning_rate, 
#                   dropout_chance=config.dropout_chance,
#                   optimizer_type=config.optimizer,
#                   reg=config.l2_reg)
#     
#     start = time.time()
#     model.fit(train, 
#               epochs=config.epochs,
#               steps_per_epoch=steps_per_epoch,
#               validation_data=valid,
#               validation_steps=validation_steps,
#               callbacks=callbacks,
#               verbose=1)
#     end = time.time()
#     print(end - start)
# =============================================================================
    
# =============================================================================
#     train_notes = "Full Train 1: (lenet model)"
#     
#     # Full Train
#     wandb.init(project="ocr",
#                name="lenet_20k_train_1",
#                notes=train_notes,
#                config={
#                        "epochs": 500,
#                        "optimizer": "adam",
#                        "learning_rate": 3e-4,
#                        "l2_reg": 1e-2,
#                        "dropout_chance": 0.5,
#                        "reduce_lr_on_plateau_monitor": "val_loss",
#                        "early_stopping_patience": 20,
#                        "reduce_lr_on_plateau_min_lr": 1e-6,
#                        "reduce_lr_on_plateau_factor":.666,
#                        "reduce_lr_on_plateau_patience": 10,
#                        })
#     config = wandb.config
#     
#     callbacks = get_callbacks(
#         early_stopping_patience=config.early_stopping_patience,
#         checkpoint_path=checkpoint_path,
#         reduce_lr_on_plateau_monitor=config.reduce_lr_on_plateau_monitor,
#         reduce_lr_on_plateau_factor=config.reduce_lr_on_plateau_factor,
#         reduce_lr_on_plateau_patience=config.reduce_lr_on_plateau_patience,
#         reduce_lr_on_plateau_min_lr=config.reduce_lr_on_plateau_min_lr
#         )
#     
# # =============================================================================
# #     model = CNN(num_classes, 
# #                 lr=config.learning_rate, 
# #                 dropout_chance=config.dropout_chance,
# #                 optimizer_type=config.optimizer,
# #                 reg=config.l2_reg)
# # =============================================================================
#     
#     model = LeNet(num_classes, 
#                   lr=config.learning_rate, 
#                   dropout_chance=config.dropout_chance,
#                   optimizer_type=config.optimizer,
#                   reg=config.l2_reg)
#     
#     start = time.time()
#     model.fit(train, 
#               epochs=config.epochs,
#               steps_per_epoch=steps_per_epoch,
#               validation_data=valid,
#               validation_steps=validation_steps,
#               callbacks=callbacks,
#               verbose=1)
#     end = time.time()
#     print(end - start)
#     
#     model.evaluate_generator(test, steps=test_steps, verbose=1)
# =============================================================================
