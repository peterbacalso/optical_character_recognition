import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
#import wandb
#from wandb.keras import WandbCallback
from data.data_loader import DataLoader
from models.crnn import CRNN

# =============================================================================
# def get_callbacks(early_stopping_patience, 
#                   reduce_lr_on_plateau_factor,
#                   reduce_lr_on_plateau_patience,
#                   reduce_lr_on_plateau_min_lr):   
# 
#     wandb_cb = WandbCallback()
#     
#     # Early Stopping
#     early_stopping = EarlyStopping(patience=early_stopping_patience, 
#                                    restore_best_weights=True)
#     
#      # Model Checkpoints
#     checkpoint = ModelCheckpoint(
#         filepath=f'checkpoints/crnn/' + \
#         'epoch.{epoch:02d}_val_loss.{val_loss:.6f}.h5', 
#         verbose=1, save_best_only=True)
#     
#     #Reduce LR on Plateau
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss',
#                                   factor=reduce_lr_on_plateau_factor,
#                                   patience=reduce_lr_on_plateau_patience, 
#                                   min_lr=reduce_lr_on_plateau_min_lr, 
#                                   verbose=1)
#     
#     return [wandb_cb, early_stopping, checkpoint, reduce_lr]
# =============================================================================

if __name__=="__main__":
    num_classes = 63 # 62 characters + 1 blank for ctc
    batch_size = 1
    seed = 23
    annotations_path="data/annotations.csv"
    images_path="data/word_images/data/"
    data = DataLoader(batch_size,  
                      annotations_path=annotations_path, 
                      images_path=images_path,
                      seed=seed)
    
    train, steps_per_epoch = data.load_text_data()
    valid, validation_steps = data.load_text_data(type='valid')
    test, test_steps = data.load_text_data(type='test')
    
    cnn_weights_path = 'checkpoints/cnn_best_weights/' + \
        'epoch.152_val_loss.0.600990.h5'
    
    model = CRNN(num_classes,
                 batch_size,
                 lr=3e-4, 
                 optimizer_type="adam",
                 reg=1e-6,
                 cnn_weights_path=cnn_weights_path)
    
    model.fit(train, 
              epochs=100,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid,
              validation_steps=validation_steps,
              verbose=1)
    
# =============================================================================
#     wandb.init(project="ocr",
#                name="rnn_10k_train_2",
#                config={
#                        "epochs": 500,
#                        "num_classes": num_classes,
#                        "batch_size": batch_size,
#                        "optimizer": "adam",
#                        "learning_rate": 3e-2,
#                        "l2_reg": 1e-6,
#                        "edit_dist_decoder": "greedy",
#                        "early_stopping_patience": 20,
#                        "reduce_lr_on_plateau_min_lr": 1e-6,
#                        "reduce_lr_on_plateau_factor":.33333,
#                        "reduce_lr_on_plateau_patience": 10,
#                        })
#     config = wandb.config
#     
#     callbacks = get_callbacks(
#         early_stopping_patience=config.early_stopping_patience,
#         reduce_lr_on_plateau_factor=config.reduce_lr_on_plateau_factor,
#         reduce_lr_on_plateau_patience=config.reduce_lr_on_plateau_patience,
#         reduce_lr_on_plateau_min_lr=config.reduce_lr_on_plateau_min_lr
#         )
# 
#     #callbacks=[WandbCallback()]
#     
#     model = CRNN(num_classes,
#                  batch_size,
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
    
    
# =============================================================================
#     results = model.evaluate_generator(test, 
#                                        steps=test_steps,
#                                        verbose=1)
# =============================================================================
    
# =============================================================================
#     # TESTING
#     def get_seq_len(data):
#         def py_get_seq_len(y_pred):
#             seq_lens = [y_pred.shape[1]]*y_pred.shape[0]
#             return [seq_lens]
#         return tf.py_function(py_get_seq_len, [data], tf.int32)
#     def index_to_label(data):
#         def py_index_to_label(values):
#             values = tf.map_fn(index_to_label_helper, values, dtype=tf.string)
#             return [values]
#         return tf.py_function(py_index_to_label, [data], tf.string)
#     def index_to_label_helper(index):
#         return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[index]
#     def decode(x):
#         labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
#         return ''.join([labels[char] for char in x[:x[-1]]])
#     import matplotlib.pyplot as plt
#     def show_img(x, scaled=True):
#         if scaled:
#             x = tf.cast((x+.5)*255.0, tf.int32)
#         x = np.squeeze(x)
#         plt.imshow(x, cmap="hot")
#         plt.show()
#     
#     for img, label in test.take(1):
#         show_img(img)
#         pred = model.predict(img)
#         print(pred.shape)
#         seq_lens = get_seq_len(pred)
#         pred_T = tf.transpose(pred, [1, 0, 2])
#         ctcOutput = \
#         tf.nn.ctc_beam_search_decoder(pred_T, seq_lens, beam_width=50, top_paths=1)
#         y_pred = ctcOutput[0][0]
#         true_values = index_to_label(y_pred.values)
#         print(true_values)
#         print(decode(label[0]))
#     
#     test_image = tf.io.read_file("data/word_images/data/455329.png")
#     test_image = tf.image.decode_png(test_image, channels=1)
#     test_image = tf.cast(test_image, tf.uint8)
#     test_image = tf.cast(test_image, tf.float16)
#     test_image = test_image/255.0 - .5
#     test_image = tf.expand_dims(test_image, 0) 
#     
#     pred = model.predict(test_image)
# 
#     pred_T = tf.transpose(pred, [1, 0, 2])
#     seq_lens = [32]*1
#     
#     inputs = tf.convert_to_tensor(pred_T, dtype=tf.float32)
#     seq_lens = tf.convert_to_tensor(seq_lens, dtype=tf.int32)
#     
#     ctcOutput = \
#     tf.nn.ctc_greedy_decoder(inputs, seq_lens, merge_repeated=True)
#     
#     decoded = ctcOutput[0][0] 
#     encodedLabelStrs = [[] for i in range(1)]
#     
#     idxDict = { b : [] for b in range(1) }
#     for (idx, idx2d) in enumerate(decoded.indices):
#         label = decoded.values[idx]
#         batchElement = idx2d[0] # index according to [b,t]
#         encodedLabelStrs[batchElement].append(label)
#         
#     charList = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
#     
#     output = [str().join([charList[c] for c in labelStr]) \
#               for labelStr in encodedLabelStrs]
#     print(output)
# =============================================================================
