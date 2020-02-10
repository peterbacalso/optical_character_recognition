import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

from PIL import Image
from scipy import io as spio
from functools import partial
from tensorflow.compat.v2.data.experimental import (
        AUTOTUNE, sample_from_datasets
        )
from sklearn.model_selection import train_test_split

import sys, os; 
sys.path.insert(0, os.path.abspath('..'));

labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class CharLoader():
    
    def __init__(self, batch_size, 
                 dataset_size=None,
                 emnist_path="characters/emnist-byclass.mat",
                 fonts_path="char_font_dataset.csv",
                 seed=23, valid_split=.2, test_split=.2, 
                 buffer_size=100000):
        emnist = spio.loadmat(emnist_path)
        
        data = emnist['dataset']

        X_train_emnist = data['train'][0,0]['images'][0,0]
        y_train_emnist = data['train'][0,0]['labels'][0,0]
        X_test_emnist = data['test'][0,0]['images'][0,0]
        y_test_emnist = data['test'][0,0]['labels'][0,0]
        
        val_start = X_train_emnist.shape[0] - X_test_emnist.shape[0]
        X_val_emnist = X_train_emnist[val_start:X_train_emnist.shape[0],:]
        y_val_emnist = y_train_emnist[val_start:X_train_emnist.shape[0]]
        X_train_emnist = X_train_emnist[0:val_start,:]
        y_train_emnist = y_train_emnist[0:val_start]
        
        if dataset_size is not None:
            X_train_emnist, _, y_train_emnist, _ = \
            train_test_split(X_train_emnist, y_train_emnist, 
                             test_size=(1-dataset_size/2/len(X_train_emnist)), 
                             random_state=seed)
            X_val_emnist, _, y_val_emnist, _ = \
            train_test_split(X_val_emnist, y_val_emnist, 
                             test_size=(1-dataset_size/2/len(X_val_emnist)), 
                             random_state=seed)
            X_test_emnist, _, y_test_emnist, _ = \
            train_test_split(X_test_emnist, y_test_emnist, 
                             test_size=(1-dataset_size/2/len(X_test_emnist)), 
                             random_state=seed)
        
        X_train_emnist, y_train_emnist = preprocess_emnist_data(
                X_train_emnist, 
                y_train_emnist)
        X_val_emnist, y_val_emnist = preprocess_emnist_data(
                X_val_emnist, 
                y_val_emnist)
        X_test_emnist, y_test_emnist = preprocess_emnist_data(
                X_test_emnist, 
                y_test_emnist)
        
        annots_raw = pd.read_csv(fonts_path)
        df, test = \
        train_test_split(annots_raw, test_size=test_split, random_state=seed)
        train, valid = \
        train_test_split(df, test_size=valid_split, random_state=seed)
        
        if dataset_size is not None:
            train = train[:dataset_size//2]
            valid = valid[:int(dataset_size//2*valid_split)]
            test = test[:int(dataset_size//2*test_split)]
        
        vf_label_to_index = np.vectorize(label_to_index)
        X_train_font, y_train_font = preprocess_font_data(
                train, vf_label_to_index)
        X_val_font, y_val_font = preprocess_font_data(
                valid, vf_label_to_index)
        X_test_font, y_test_font = preprocess_font_data(
                test, vf_label_to_index)
        
        X_train = np.concatenate((X_train_emnist, X_train_font))
        y_train = np.concatenate((y_train_emnist, y_train_font))
# =============================================================================
#         X_train = X_train_emnist.numpy()
#         y_train = y_train_emnist.numpy()
# =============================================================================
        
        X_val = np.concatenate((X_val_emnist, X_val_font))
        y_val = np.concatenate((y_val_emnist, y_val_font))
# =============================================================================
#         X_val = X_val_emnist.numpy()
#         y_val = y_val_emnist.numpy()
# =============================================================================
        
        X_test = np.concatenate((X_test_emnist, X_test_font))
        y_test = np.concatenate((y_test_emnist, y_test_font))
        
        if dataset_size is not None:
            if dataset_size < 10:
                print('TRAIN SAMPLE')
                show_img(X_train[0], scaled=False)
                print('X_train Label:', y_train[0])
                print("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[y_train[0]])
                print('VALID SAMPLE')
                show_img(X_val[0], scaled=False)
                print('X_train Label:', y_val[0])
                print("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[y_val[0]])
                print('TEST SAMPLE')
                show_img(X_test[0], scaled=False)
                print('X_train Label:', y_test[0])
                print("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[y_test[0]])
                print()
        
# =============================================================================
#         # check class distribution of subsample
#         unique, counts = np.unique(y_train, return_counts=True)
#         print(len(unique))
#         print(np.asarray((unique, counts)).T)
# =============================================================================
        
        augmenter = iaa.Sequential([
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-4, 4)
            )
        ], random_order=True)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.seed = seed
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.augmenter = augmenter
        
    def load_data(self, type="train", img_size=32, onehot=True, augment=False):
        if type=="test":
            X = self.X_test
            y = self.y_test
        elif type=="valid":
            X = self.X_val
            y = self.y_val
        else:
            X = self.X_train
            y = self.y_train
        
        steps_per_gen = np.ceil(len(X)/self.batch_size)
        classes = np.unique(y)
        num_classes = len(classes)
        
        if type=="test" or type=="valid":
            data = tf.data.Dataset.from_tensor_slices((X, y)).cache()
            data = data.shuffle(buffer_size=self.buffer_size, 
                                seed=self.seed).repeat()
        else:
            # Oversampling
            datasets = []
            for i in range(num_classes):
                y_series = pd.Series(np.squeeze(y))
                indices = np.array(y_series[y_series==classes[i]].index)
                #print(indices)
                sub_X = X[indices]
                sub_y = y[indices]
                sub = tf.data.Dataset\
                .from_tensor_slices((sub_X, sub_y)).cache()
                sub = sub.shuffle(buffer_size=self.buffer_size, 
                                  seed=self.seed).repeat()
                datasets.append(sub)
            sampling_weights = np.ones(num_classes)*(1./num_classes)
            data = sample_from_datasets(datasets, 
                                        weights=sampling_weights, 
                                        seed=self.seed)
        if onehot:
            data = data.map(partial(one_hot, num_classes=num_classes),
                            num_parallel_calls=AUTOTUNE)
        if augment:
            data = data.map(
                    partial(augment_img, 
                            augmenter=self.augmenter),
                    num_parallel_calls=AUTOTUNE)

        data = data.map(standard_scaler, num_parallel_calls=AUTOTUNE)
    
        data = data.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        
        return data, steps_per_gen, num_classes
    

def preprocess_emnist_data(X, y):
    X = X.reshape( (X.shape[0], 28, 28), order='F')
    X = np.expand_dims(X, axis=-1) 
    X = tf.image.resize(X, size=(32,32), preserve_aspect_ratio=False)
    y = tf.reshape(y, [len(y)])
    return X, y

def preprocess_font_data(data, vf_label_to_index):
    X = data[[str(i) for i in range(1024)]].to_numpy()
    X = X.reshape((X.shape[0], 32, 32), order='C')
    X = np.expand_dims(X, axis=-1) 
    X = tf.image.resize(X, size=(32,32), preserve_aspect_ratio=False)
    y = data['annotation'].to_numpy()
    y = vf_label_to_index(y)
    return X, y

def augment_img(images, output, augmenter):
    images = tf.cast(images, tf.uint8)
    images = tf.numpy_function(augmenter.augment_image, [images], tf.uint8)
    images = tf.reshape(images, shape=(32,32,1))
    return images, output
          
def one_hot(img, outputs, num_classes):
    new_outputs = tf.one_hot(outputs, num_classes)
    return img, new_outputs

def standard_scaler(images, index_list):
    images = tf.cast(images, tf.float16)
    images = images/255.0 - .5
    return images, index_list

def show_img(x, scaled=True):
    if scaled:
        x = tf.cast((x+.5)*255.0, tf.int32)
    x = np.squeeze(x)
    plt.imshow(x, cmap="hot")
    plt.show()
    
def load_img(file):
    im = Image.open(file)
    return im

def label_to_index(label):
    labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return labels.index(label)

if __name__=="__main__":
    data = CharLoader(batch_size=2, dataset_size=10)
    #valid, valid_steps, num_classes = data.load_data(type='valid', onehot=False)
    
    train, _, _ = data.load_data(onehot=False, augment=True)
    for img, label in train.take(5):
        print(img.shape)
        print(label.shape)
        #print(img[0])
        show_img(img[0])
        print(label[0])
        print("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[label[0]])
    
    val, _, _ = data.load_data(type="valid", onehot=False)
    for img, label in val.take(5):
        print(img.shape)
        print(label.shape)
        #print(img[0])
        show_img(img[0])
        print(label[0])
        print("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[label[0]])
    

