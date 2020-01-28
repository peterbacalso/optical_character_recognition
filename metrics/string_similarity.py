import tensorflow as tf
import numpy as np
from similarity.normalized_levenshtein import NormalizedLevenshtein
from tensorflow.keras.metrics import Metric


labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
norm_lev = NormalizedLevenshtein()

class LevenshteinMetric(Metric):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.levenshtein_distance_fn = levenshtein_distance_fn
        self.batch_size = batch_size
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.levenshtein_distance_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(self.batch_size, tf.float32))
        #self.count.assign_add(tf.cast(len(y_true), tf.float32))
    def result(self):
        return self.total / self.count
    def get_config(self):
        base_config = super().get_config()
        return base_config

def levenshtein_distance_fn(y_true, y_pred):
    '''
    inputs:
    y_true - ground truth string value of image
    y_pred - predicted string value of dimension
             [batch_size, timestep, n_classes]
    
    returns:
    levenshtein distance between y_true and y_pred
    '''    
    sparse_tensor = to_sparse(y_true)
    true_decoded = tf.sparse.SparseTensor(sparse_tensor[0], 
                                          sparse_tensor[1], 
                                          sparse_tensor[2])
    
    seq_lens = get_seq_len(y_pred)
    
    y_pred_T = tf.transpose(y_pred, [1, 0, 2])
    ctcOutput = \
    tf.nn.ctc_greedy_decoder(y_pred_T, seq_lens, merge_repeated=True)
    pred_decoded = ctcOutput[0][0] 
    
    lev_dist = tf.edit_distance(true_decoded, pred_decoded, normalize=True)
    
    return lev_dist

@tf.function(autograph=False)
def get_seq_len(data):
    def py_get_seq_len(y_pred):
        seq_lens = [y_pred.shape[1]]*y_pred.shape[0]
        return [seq_lens]
    return tf.py_function(py_get_seq_len, [data], tf.int32)

@tf.function(autograph=False)
def to_sparse(data):
    def py_to_sparse(y_true):
        indices = []
        values = []
        shape = [y_true.shape[0], y_true.shape[1]]
        
        # go over all texts
        for (batchElement, index_list) in enumerate(y_true.numpy()):
            # convert to string of label (i.e. class-ids)
            labels = index_list[:index_list[-1]]
            # put each label into sparse tensor
            for (i, label) in enumerate(labels):
                indices.append([batchElement, i])
                values.append(label)
                    
        return (indices, values, shape)
    return tf.py_function(py_to_sparse, [data], (tf.int64,tf.int64,tf.int64))

if __name__=="__main__":
    charList = "ab" 
    batchSize=2
    # (5,2,3) [max_time, batch_size, num_classes]
    a = np.array([[[1,0,0],[0,1,0]],
                  [[0,0,1],[0,0,1]],
                  [[0,0,1],[0,0,1]],
                  [[0,0,1],[0,0,1]],
                  [[0,0,1],[1,0,0]]])
    b = np.array([5,5])
    
    c = np.array([[[0,0,1],[0,1,0]],
                  [[0,0,1],[0,0,1]],
                  [[0,1,0],[0,0,1]],
                  [[0,0,1],[0,0,1]],
                  [[0,1,0],[1,0,0]]])
    d = np.array([5,5])
    
    inputs = tf.convert_to_tensor(a, dtype=tf.float32)
    seq_lens = tf.convert_to_tensor(b, dtype=tf.int32)
    
    inputs2 = tf.convert_to_tensor(c, dtype=tf.float32)
    seq_lens2 = tf.convert_to_tensor(d, dtype=tf.int32)
    
    ctcOutput = \
    tf.nn.ctc_greedy_decoder(inputs, seq_lens, merge_repeated=True)
    
    ctcOutput2 = \
    tf.nn.ctc_greedy_decoder(inputs2, seq_lens2, merge_repeated=True)
# =============================================================================
#     ctcOutput = \
#     tf.nn.ctc_beam_search_decoder(inputs,seq_lens,beam_width=100,top_paths=1)
# =============================================================================
    
    decoded = ctcOutput[0][0] 
    neg_sum_logits = ctcOutput[1]
    
    decoded2 = ctcOutput2[0][0] 
    
    l_d = tf.edit_distance(decoded, decoded2, normalize=True)
    
    encodedLabelStrs = [[] for i in range(batchSize)]
    encodedLabelStrs2 = [[] for i in range(batchSize)]
    
    print(l_d)
# =============================================================================
#     print(decoded.indices)
#     print(decoded.values)
# =============================================================================
    
    idxDict = { b : [] for b in range(batchSize) }
    for (idx, idx2d) in enumerate(decoded.indices):
        label = decoded.values[idx]
        batchElement = idx2d[0] # index according to [b,t]
        encodedLabelStrs[batchElement].append(label)
    
    output = [str().join([charList[c] for c in labelStr]) \
              for labelStr in encodedLabelStrs]
    print(output)
    
    idxDict2 = { b : [] for b in range(batchSize) }
    for (idx2, idx2d2) in enumerate(decoded2.indices):
        label2 = decoded2.values[idx2]
        batchElement2 = idx2d2[0] # index according to [b,t]
        encodedLabelStrs2[batchElement2].append(label2)
    
    output2 = [str().join([charList[c] for c in labelStr]) \
              for labelStr in encodedLabelStrs2]
    print(output2)
    
    
# =============================================================================
#     y_pred = tf.constant(["add d ff f", "hhee lll lo"])
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
