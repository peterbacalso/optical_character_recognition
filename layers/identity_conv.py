import tensorflow as tf
import numpy as np

# for testing
import matplotlib.pyplot as plt
import sys, os; 
sys.path.insert(0, os.path.abspath('..'));
from models.cnn import CNN
from models.lenet import LeNet

def identity_conv(images, patch_size, stride=2, rate=1, padding='VALID'):
    '''
    Only slides kernel accross width for this implementation
    
    images  - (batch_size, image_height, image_width, channels)
    patches - (batch_size, num_windows, image_height*window_width*channels)
    '''
    sizes = get_kernel(patch_size)
    strides = get_kernel(stride)
    rates = get_kernel(rate)
    patches = tf.image.extract_patches(images=images,
                                       sizes=sizes,
                                       strides=strides,
                                       rates=rates,
                                       padding=padding)
    patches = tf.squeeze(patches, axis=1)
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

def show_img(x, scaled=True):
    if scaled:
        x = tf.cast((x+.5)*255.0, tf.int32)
    x = np.squeeze(x)
    plt.imshow(x, cmap="hot")
    plt.show()

if __name__=="__main__":
# =============================================================================
#     h = 128
#     w = 32
#     img1 = [[[[x * 10 + y + 1] for y in range(h)] for x in range(w)]]
#     img2 = [[[[x * 10 + y + 1] for y in range(h)] for x in range(w)]]
#     
#     img1 = np.array(img1)
#     img2 = np.array(img2)
#     img = np.concatenate((img1, img2), axis=0)
#     print(img.shape)
#     
#     patches = identity_conv(img, patch_size=(32,16))
#     print(patches.shape)
# =============================================================================
    
    #####################################################################
    
    # test image ("midbrains")
    test_image = tf.io.read_file("../data/word_images/data/684563.png")
    # test image ("detestable")
    #test_image = tf.io.read_file("../data/word_images/data/100026.png")
    test_image = tf.image.decode_png(test_image, channels=1)
    test_image = tf.cast(test_image, tf.uint8)
    test_image = tf.cast(test_image, tf.float16)
    test_image = test_image/255.0 - .5
    test_image = tf.expand_dims(test_image, 0) 
    
    print(test_image.shape)
    patches = identity_conv(test_image, patch_size=(32,16), stride=2)
    print('patches shape', patches.shape)
    patches = tf.reshape(patches, (patches.shape[1],32,16,1))
    print('patches reshape', patches.shape)
    
# =============================================================================
#     for i in range(patches.shape[0]):
#         show_img(patches[i])
# =============================================================================
        
    cnn = CNN(62, reg=3e-4, compile_model=True)
# =============================================================================
#     cnn_weights_path = '../checkpoints/cnn_best_weights/' + \
#         'epoch.158_val_loss.0.628467.h5'
# =============================================================================
    cnn_weights_path = '../checkpoints/cnn_best_weights/' + \
        'epoch.152_val_loss.0.600990.h5'     
    cnn.load_weights(cnn_weights_path)
    #cnn = Model(inputs=cnn.inputs, outputs=cnn.layers[-2].output)
    
# =============================================================================
#     lenet = LeNet(62, reg=3e-4, compile_model=True)
#     lenet_weights_path = '../checkpoints/cnn_best_weights/' + \
#         'lenet_epoch.276_val_loss.1.301563.h5'
#     lenet.load_weights(lenet_weights_path)
# =============================================================================
    
    probs = []
    word = []
# =============================================================================
#     # padding test
#     for i in range(patches.shape[0]):
#         t = patches[i]
#         paddings = tf.constant([[0,0],[8,8],[0,0]])    
#         t_unscaled = tf.cast((t+.5)*255.0, tf.int32)
#         padded_img = tf.pad(t_unscaled, paddings, "CONSTANT")        
#         print(padded_img.shape)
#         show_img(padded_img, scaled=False)
#         padded_img = tf.cast(padded_img, tf.float32)
#         t_scaled = padded_img/255.0 - .5
#         t_scaled = tf.expand_dims(t_scaled, axis=0)
#         index = cnn.predict(t_scaled)
#         probs.append(int(np.max(index)*100))
#         print(probs)
#         index = np.argmax(index)
#         word.append("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[index])
#         print(word)
# =============================================================================
    
    # repeat test
    for i in range(patches.shape[0]):
        t = patches[i]   
        t_unscaled = tf.cast((t+.5)*255.0, tf.int32)
        #repeat_img = np.repeat(t_unscaled,2, axis=1)
        
        repeat_img = tf.keras.backend.repeat_elements(t_unscaled, 2, axis=1)
        
        print(repeat_img.shape)
        show_img(repeat_img, scaled=False)
        repeat_img = tf.cast(repeat_img, tf.float32)
        t_scaled = repeat_img/255.0 - .5
        t_scaled = tf.expand_dims(t_scaled, axis=0)
        index = cnn.predict(t_scaled)
        pred_prob = int(np.max(index)*100)
        print(pred_prob)
        probs.append(pred_prob)
        index = np.argmax(index)
        pred = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[index]
        print(pred)
        word.append(pred)
        break
    print(probs)    
    print(word)