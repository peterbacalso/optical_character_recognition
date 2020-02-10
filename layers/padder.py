import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def padder(t, pad_size=8, axis=2):
    '''
    input  - [batch_size, img_height, img_width, channels]
    output - [batch_size, img_height, padded_img_width, channels]
    '''
    paddings = []
    for i in range(len(t.shape)):
        if i == axis:
            paddings.append([pad_size,pad_size])
        else:
            paddings.append([0,0])
    paddings = tf.constant(paddings)
    t_unscaled = tf.cast((t+.5)*255.0, tf.int32)
    padded_img = tf.pad(t_unscaled, paddings, "CONSTANT")
    padded_img = tf.cast(padded_img, tf.float32)
    t_scaled = padded_img/255.0 - .5
    #show_img(padded_img)
    return t_scaled

def show_img(x, scaled=True):
    if scaled:
        x = tf.cast((x+.5)*255.0, tf.int32)
        #x = (x+.5)*255.0
    x = tf.squeeze(x)
    tf.print(x)
