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
        return tf.reduce_mean(tf.keras.backend.ctc_batch_cost(
            y_true=labels,
            y_pred=y_pred,
            input_length=self.logit_length,
            label_length=label_length
        ))
# =============================================================================
#         return tf.reduce_mean(tf.nn.ctc_loss(
#             labels=labels, 
#             logits=y_pred, 
#             label_length=label_length,
#             logit_length=self.logit_length,
#             logits_time_major=self.logits_time_major,
#             blank_index=self.blank_index
#         ))
# =============================================================================
    def get_config(self):
        base_config = super().get_config()
        return base_config
    
@tf.function(autograph=False)
def get_labels(data):
    return tf.py_function(py_get_labels2, [data], (tf.int64, tf.int32))

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

def py_get_labels2(y_true):
    y_true = y_true.numpy().astype(np.int32)
    # the labels length is set at the last index of every y_true
    label_length = \
    np.array([i[-1] for i in y_true]).astype(np.int32)
    labels = np.array(y_true[:,:-1])
    #labels = np.expand_dims(labels, axis=0)
    label_length = np.expand_dims(label_length, axis=1)
    #tf.print('y_true shape', labels.shape)
    #tf.print('label_length shape', label_length.shape)
    return labels, label_length

def labels_to_index_list(text, max_str_len):
    classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
    initial_text_len = len(text)
    text = text + " "*(max_str_len-initial_text_len)
    index_list = [classes.index(char) for char in text]
    index_list.append(initial_text_len)
    return np.array(index_list, dtype=object)

def labels_to_index_list_with_blank(text, max_str_len):
    classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
    formatted_text = ""
    prev_char = ''
    for char in text:
        if prev_char == char:
            formatted_text = formatted_text + ' '
        formatted_text = formatted_text + char
        prev_char = char
    formatted_text_len = len(formatted_text)
    formatted_text = formatted_text + " "*(max_str_len-formatted_text_len)
    index_list = [classes.index(char) for char in formatted_text]
    index_list.append(formatted_text_len)
    return np.array(index_list, dtype=object)



def logitify(labels, timesteps):
    logits = np.ones((1,timesteps,63))
    for index, label in enumerate(labels):
        s = np.random.random_sample((1,63))
        s[0][label] = 1000
        s = s / np.sum(s)
        print(np.argmax(s), np.max(s))
        filler = np.ones((1,63))
        concat_order = []
        for i in range(timesteps):
            concat_order.append(s) if i == index else concat_order.append(filler)
        broadcast = np.concatenate(concat_order, axis=0)
        logits = logits*broadcast
    return logits

def indexify(word, max_str_len=32, with_blank=False):
    if with_blank:
        index_list = labels_to_index_list_with_blank(word, max_str_len)
    else:
        index_list = labels_to_index_list(word, max_str_len)
    index_list = tf.convert_to_tensor(index_list, dtype=tf.int32)
    index_list = tf.expand_dims(index_list, axis=0)
    labels, label_length = py_get_labels2(index_list)
    return labels, label_length

@tf.function(autograph=False)
def index_to_label(data):
    def py_index_to_label(values):
        values = tf.map_fn(index_to_label_helper, values, dtype=tf.string)
        return [values]
    return tf.py_function(py_index_to_label, [data], tf.string)

def index_to_label_helper(index):
    classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
    return classes[index]

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
    word = "hello"
    correct = "hel lo" # correct
    dup = "hhel lo" # duplicate test
    dup2 = "hheel loo"
    one_l = "helo"
    missing_h = "ello" # missing head
    missing_t = "hell" # missing tail
    wrong = "bob" # wrong word
    
    
    
    labels, label_length = indexify(word)  # correct 
    correct_labels, _ = indexify(correct)  # correct 
    dup_labels, _ = indexify(dup) # dup
    dup2_labels, _ = indexify(dup2) # dup2
    one_l_labels, _ = indexify(one_l) # one_l
    missing_h_labels, _ = indexify(missing_h) # missing head
    missing_t_labels, _ = indexify(missing_t) # missing head
    wrong_labels, _ = indexify(wrong) # wrong
    
    print('labels', labels)
    
    #timesteps = 32
    timesteps = len(labels[0])
    #timesteps = label_length[0] # for batch cost

# =============================================================================
#     # random
#     s = np.random.random_sample((63,))
#     s = s / np.sum(s)
#     logits = np.ones((1,32,63))
#     logits = logits * s
# =============================================================================
    
    # index has highest prob
    #logits = logitify(correct_labels[0], timesteps) # correct
    #logits = logitify(dup_labels[0], timesteps) # dup
    #logits = logitify(dup2_labels[0], timesteps) # dup2
    #logits = logitify(one_l_labels[0], timesteps) # one_l
    #logits = logitify(missing_h_labels[0], timesteps) # missing head
    logits = logitify(missing_t_labels[0], timesteps) # missing tail
    #logits = logitify(wrong_labels[0], timesteps) # wrong
    
    # check logits are correct
    seq_lens = get_seq_len(logits)
    y_pred_T = tf.transpose(logits, [1, 0, 2])
    y_pred_T = tf.cast(y_pred_T, tf.float32)
    ctcOutput = \
    tf.nn.ctc_greedy_decoder(y_pred_T, seq_lens, merge_repeated=True)
# =============================================================================
#     ctcOutput = \
#     tf.nn.ctc_beam_search_decoder(y_pred_T, seq_lens, beam_width=50, top_paths=1)
# =============================================================================
    pred = ctcOutput[0][0] 
    pred_values = index_to_label(pred.values)
    tf.print('PRED', pred_values, summarize=32)
    
    #logit_length = np.array([32])
    logit_length = np.array([[32]]) # for batch cost
    #label_length = np.expand_dims(label_length, axis=1) # for batch cost
    
    labels = tf.convert_to_tensor(labels)
    label_length = tf.convert_to_tensor(label_length)
    logits = tf.convert_to_tensor(logits)
    logits = tf.cast(logits, tf.float32)
    logit_length = tf.convert_to_tensor(logit_length)
    print('labels', labels)
    print('label_length', label_length)
    print('logits shape', logits.shape)
    print('logit_length', logit_length)
    
# =============================================================================
#     loss = tf.nn.ctc_loss(
#             labels=labels, 
#             logits=logits, 
#             label_length=label_length,
#             logit_length=logit_length,
#             logits_time_major=False,
#             blank_index=62
#         )
# =============================================================================
    
    loss = tf.keras.backend.ctc_batch_cost(
        y_true=labels,
        y_pred=logits,
        input_length=logit_length,
        label_length=label_length
    )
    
    print(loss)
    
    
    


