import tensorflow as tf

# =============================================================================
# def ctc_loss(y_true, y_pred, sample_weight, labels, label_length, 
#              input_length, search_strat="default"):
#     # todo: add option ctc beam search loss decoder
#     return tf.nn.ctc_loss(labels, y_pred, label_length, 
#                           input_length, logits_time_major=False)
# =============================================================================
    
def ctc_loss(args):
    labels, y_pred, label_length, input_length = args
    # todo: add option ctc beam search loss decoder
    return tf.nn.ctc_loss(labels, y_pred, label_length, 
                          input_length, logits_time_major=False)

# =============================================================================
# def ctc_loss(args):
#     labels, y_pred, label_length, input_length = args
#     return tf.keras.backend.ctc_batch_cost(
#             labels, y_pred, input_length, label_length)
# =============================================================================
