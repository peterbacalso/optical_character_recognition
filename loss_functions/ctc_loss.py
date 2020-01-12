import tensorflow as tf
import numpy as np

class CTCLoss(tf.losses.Loss):
    def __init__(self, logit_length, blank_index=-1, 
                 logits_time_major=False):
        super(CTCLoss, self).__init__()
        self.logit_length = tf.convert_to_tensor(logit_length)
        self.blank_index = blank_index
        self.logits_time_major = logits_time_major

    def call(self, y_true, y_pred):
        labels, label_length = get_labels(y_true)
        return tf.reduce_mean(tf.nn.ctc_loss(
            labels=labels, 
            logits=y_pred, 
            label_length=label_length,
            logit_length=self.logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index
        ))
    
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
        return labels, label_length
    return tf.py_function(py_get_labels, [data], (tf.int64, tf.int32))