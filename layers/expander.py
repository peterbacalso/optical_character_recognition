import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def expander(t):
    '''
    input  - [batch_size, img_height, img_width, channels]
    output - [batch_size, img_height, expanded_img_width, channels]
    '''
    t_unscaled = tf.cast((t+.5)*255.0, tf.int32)
    repeat_img = tf.keras.backend.repeat_elements(t_unscaled, 2, axis=2)
    repeat_img = tf.cast(repeat_img, tf.float32)
    t_scaled = repeat_img/255.0 - .5

    return t_scaled



