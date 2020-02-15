import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from data.data_loader import DataLoader
from models.crnn import CRNN

if __name__=="__main__":
    num_classes = 63 # 62 characters + 1 blank for ctc
    batch_size = 64
    seed = 23
    annotations_path="data/annotations.csv"
    images_path="data/word_images/data/"
    checkpoint_path = "checkpoints/crnn/"
    data = DataLoader(batch_size,  
                      annotations_path=annotations_path, 
                      images_path=images_path,
                      seed=seed)
    
    test, test_steps = data.load_text_data(type='test')
    
    weights_path = 'checkpoints/crnn_best_weights/crnn_best.h5'
    
    crnn = CRNN(num_classes, batch_size)
    crnn.load_weights(weights_path)
    
    results = crnn.evaluate_generator(test, 
                                       steps=test_steps,
                                       verbose=1)
    
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
#     for imgs, label in test.take(1):
#         img = imgs[0]
#         show_img(img)
#         img = tf.expand_dims(img, axis=0)
#         pred = crnn.predict(img)
#         print(pred.shape)
#         seq_lens = get_seq_len(pred)
#         pred_T = tf.transpose(pred, [1, 0, 2])
#         ctcOutput = \
#         tf.nn.ctc_greedy_decoder(pred_T, seq_lens, merge_repeated=True)
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
#     pred = crnn.predict(test_image)
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
