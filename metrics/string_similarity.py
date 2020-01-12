import tensorflow as tf
import numpy as np
from similarity.normalized_levenshtein import NormalizedLevenshtein
from tensorflow.keras.metrics import Metric


labels = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
norm_lev = NormalizedLevenshtein()

class LevenshteinMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.levenshtein_distance_fn = levenshtein_distance_fn
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("total", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.levenshtein_distance_fn(y_true, y_pred)
# =============================================================================
#         self.total.assign_add(tf.reduce_sum(metric))
#         self.count.assign_add(tf.size(y_true))
# =============================================================================
    def result(self):
        return self.total / self.count

def levenshtein_distance_fn(y_true, y_pred):
    '''
    inputs:
    y_true - ground truth string value of image
    args   - array of [y_pred, ctc_loss] where y_pred 
             is [batch_size, timestep, n_classes]
    
    returns:
    levenshtein distance between y_true and predicted string
    '''    
    index_tensor = tf.math.argmax(y_pred, axis=2)
    pred_text_raw = tf.map_fn(convert_to_text, index_tensor, dtype=tf.string)
    tf.print(pred_text_raw)
# =============================================================================
#     pred_text = tf.map_fn(remove_blanks_and_duplicates, pred_text_raw)
#     
#     true_text = get_labels(y_true)
#     
#     true_text = tf.expand_dims(true_text, 0)
#     pred_text = tf.expand_dims(pred_text, 0)
#     all_text = tf.concat([true_text, pred_text], axis=0)
#     all_text = tf.transpose(all_text)
#     
#     tf.print(tf.map_fn(calc_norm_lev, all_text, dtype=tf.float32))
# =============================================================================
    return 1

@tf.function
def convert_to_text(index_list):
    '''
    index_tensor - [timestep]
    '''
    labels_list = tf.map_fn(index_to_label, index_list, dtype=tf.string)
    return tf.strings.reduce_join(labels_list, axis=0)

@tf.function(autograph=False)
def index_to_label(data):
    def py_index_to_label(index):
        return labels[index] if index > 0 else " "
    return tf.py_function(py_index_to_label, [data], tf.string)

def remove_blanks_and_duplicates(text):
    decoded_text = text.numpy().decode("utf-8").split()
    return "".join(["".join(sorted(set(x), key=x.index)) 
                    for x in decoded_text])
    
def calc_norm_lev(args):
    y_true, y_pred = args
    
    y_true = y_true.numpy().decode("utf-8")
    y_pred = y_pred.numpy().decode("utf-8")
    
    return norm_lev.distance(y_true, y_pred)

@tf.function(autograph=False)
def get_labels(data):
    def py_get_labels(y_true):
        y_true = y_true.numpy().astype(np.int32)
        # the labels length is set at the last index of every y_true
        label_length = \
        np.array([i[-1] for i in y_true]).astype(np.int32)
        labels = np.zeros(
            (len(label_length), np.max(label_length))).astype(np.int32)
        for nxd, i in enumerate(y_true):
            labels[nxd, :i[-1]] = i[:i[-1]].astype(np.int32)
        return labels
    return tf.py_function(py_get_labels, [data], (tf.int64, tf.int32))

if __name__=="__main__":
    y_pred = tf.constant(["add d ff f", "hhee lll lo"])
# =============================================================================
#     y_pred = tf.map_fn(remove_blanks_and_duplicates, y_pred)
#     y_pred = tf.expand_dims(y_pred, 0)
#     
#     y_true = tf.constant(["and", "hello"])
#     y_true = tf.expand_dims(y_true, 0)
#     
#     y_all = tf.concat([y_true, y_pred], axis=0)
#     y_all = tf.transpose(y_all)
#     score = tf.map_fn(calc_norm_lev, y_all, dtype=tf.float32)
#     print(score)
# =============================================================================
    
    
    
# =============================================================================
#     str_1 = 'supe'
#     str_2 = 'super'
#     
#     start_1 = time.time()
#     hypothesis = list(str_1)
#     truth = list(str_2)
#     h1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3]],
#                      hypothesis,
#                      [1,1,1])
#     t1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,1], [0,0,3],[0,0,4]],
#                      truth,
#                      [1,1,1])
#     print(tf.edit_distance(h1, t1, normalize=True))
#     end_1 = time.time()
#     print(end_1-start_1)
#     
#     norm_lev = NormalizedLevenshtein()
#     start_2 = time.time()
#     print(norm_lev.distance(str_1, str_2))
#     end_2 = time.time()
#     print(end_2-start_2)
# =============================================================================
