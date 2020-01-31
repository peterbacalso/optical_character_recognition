import tensorflow as tf
import numpy as np

def identity_conv(images, patch_size, stride=2, rate=1, padding='VALID'):
    '''
    Only slides kernel accross width for this implementation
    
    images  - (batch_size, image_height, image_width, channels)
    patches - (batch_size, num_windows, image_height, window_width, channels)
    '''
    sizes = get_kernel(patch_size)
    strides = get_kernel(stride)
    rates = get_kernel(rate)
    patches = tf.image.extract_patches(images=images,
                                       sizes=sizes,
                                       strides=strides,
                                       rates=rates,
                                       padding=padding)

    batch_size = patches.shape[0]
    num_windows = patches.shape[2]
    patches = tf.squeeze(patches, axis=1)
    shape = [batch_size, num_windows, sizes[1], sizes[2], images.shape[3]]
    patches = tf.reshape(patches, shape)
    return patches
    
def get_kernel(kernel_size):
    if isinstance(kernel_size, int):
        size = [1, kernel_size, kernel_size, 1]
    elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) <= 2:
        if len(kernel_size) == 1:
            size = [1, kernel_size[0], kernel_size[0], 1]
        else:
            size = [1, kernel_size[0], kernel_size[1], 1]
    else:
        raise ValueError('Invalid kernel_size. Must be int or list/tuple with length 1 or 2')
    return size

if __name__=="__main__":
    h = 128
    w = 32
    img1 = [[[[x * 10 + y + 1] for y in range(h)] for x in range(w)]]
    img2 = [[[[x * 10 + y + 1] for y in range(h)] for x in range(w)]]
    
    img1 = np.array(img1)
    img2 = np.array(img2)
    img = np.concatenate((img1, img2), axis=0)
    print(img.shape)
    
    patches = identity_conv(img, patch_size=(32,16))
    print(patches.shape)
