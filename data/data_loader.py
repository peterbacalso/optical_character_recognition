import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import io as spio
from functools import partial
from tensorflow.compat.v2.data.experimental import (
        AUTOTUNE, sample_from_datasets
        )
from sklearn.model_selection import train_test_split

import sys, os; 
sys.path.insert(0, os.path.abspath('..'));

BUFFER_SIZE = 100000

labels = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class DataLoader():
    
    def __init__(self, batch_size, 
                 annotations_path="annotations.csv",
                 images_path="word_images/data/",
                 seed=23, valid_split=.2, test_split=.2):
        annots_raw = pd.read_csv(annotations_path)
        annots_raw["index"] = annots_raw["index"].map(
                lambda x: '{}{}.png'.format(images_path, x))
        annots_raw.rename(columns={"index": "path"}, inplace =True)
        annots_raw['path'] = \
        annots_raw['path'].astype('|S').str.decode("utf-8")
        annots_raw['annotation'] = \
        annots_raw['annotation'].astype('|S').str.decode("utf-8")
        max_str_len = max(annots_raw['annotation'].str.len())
        
        df, test = train_test_split(annots_raw, test_size=test_split, 
                                       random_state=seed)
        train, valid = train_test_split(df, test_size=valid_split,
                                              random_state=seed)
        train = train[:int(len(train)*.05)]
        valid = valid[:int(len(valid)*.05)]
        test = test[:int(len(test)*.05)]
        
        vf = np.vectorize(partial(labels_to_index_list, 
                                  max_str_len=max_str_len))
        
        y_train_list = train['annotation'].to_numpy()
        y_train = vf(y_train_list)
        y_train = np.array([x.astype('int32') for x in y_train])
        
        y_valid_list = valid['annotation'].to_numpy()
        y_valid = vf(y_valid_list)
        y_valid = np.array([x.astype('int32') for x in y_valid])
        
        y_test_list = test['annotation'].to_numpy()
        y_test = vf(y_test_list)
        y_test = np.array([x.astype('int32') for x in y_test])
        
        self.x_train = train["path"]
        self.y_train = y_train
        self.x_valid = valid["path"]
        self.y_valid = y_valid
        self.x_test = test["path"]
        self.y_test = y_test
        self.seed=seed
        self.batch_size=batch_size
        self.max_str_len = max_str_len
        
    def load_text_data(self, type="train"):
        if type=="test":
            paths = self.x_test
            index_list = self.y_test
        elif type=="valid":
            paths = self.x_valid
            index_list = self.y_valid
        else:
            paths = self.x_train
            index_list = self.y_train
        steps_per_epoch = np.ceil(len(paths)/self.batch_size)
        input_targets = make_ds(paths, index_list, 
                                seed=self.seed)
        imgs_targets = input_targets.map(load_image,
                                         num_parallel_calls=AUTOTUNE)
        imgs_targets = imgs_targets.map(standard_scaler, 
                                        num_parallel_calls=AUTOTUNE)
        ds = imgs_targets.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        return ds, steps_per_epoch
    
def standard_scaler(images, index_list):
    images = tf.cast(images, tf.float16)
    images = images/255.0 - .5
    return images, index_list

def make_ds(paths, index_list, seed):
    ds = tf.data.Dataset.from_tensor_slices((paths, index_list)).cache()
    ds = ds.shuffle(BUFFER_SIZE, seed=seed).repeat()
    return ds

def load_image(paths, index_list, channels=1):
    images = tf.io.read_file(paths)
    images = tf.image.decode_jpeg(images, channels=channels)
    images = tf.cast(images, tf.uint8)
    return images, index_list
    
    
    
# Emnist loader
        
def load_emnist(batch_size, onehot=True, 
                valid_split=.2, data_path="characters/emnist-byclass.mat"):
    emnist = spio.loadmat(data_path)
    num_classes = 62
    buffer_size=100000
    seed=23
    
    # load training dataset
    x_raw = emnist["dataset"][0][0][0][0][0][0]
    x_raw = x_raw.astype(np.float32)
    # load training labels
    y_raw = emnist["dataset"][0][0][0][0][0][1]
    y_raw = y_raw.flatten()
    
    x_train, x_valid, y_train, y_valid = \
    train_test_split(x_raw, y_raw, test_size=valid_split)
    
    # load test dataset
    x_test = emnist["dataset"][0][0][1][0][0][0]
    x_test = x_test.astype(np.float32)
    # load test labels
    y_test = emnist["dataset"][0][0][1][0][0][1]
    y_test = y_test.flatten()
    
    # Preprocess input data, reshape using matlab order
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1, order="A")
    x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1, order="A")
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1, order="A")
    
# =============================================================================
#     # train on small portion of data
#     x_train = x_train[:20]
#     y_train = y_train[:20]
# =============================================================================
    
    steps_per_epoch = np.ceil(len(x_train)/batch_size)
    validation_steps = np.ceil(len(x_valid)/batch_size)
    
    # Oversampling
    train_datasets = []
    for i in range(num_classes):
        sub_x_train = x_train[y_train==i]
        sub_y_train = y_train[y_train==i]
        sub_train = tf.data.Dataset\
        .from_tensor_slices((sub_x_train, sub_y_train)).cache()
        sub_train = sub_train.shuffle(buffer_size=buffer_size, seed=seed).repeat()
        train_datasets.append(sub_train)
    sampling_weights = np.ones(num_classes)*(1./num_classes)
    train = sample_from_datasets(train_datasets, 
                                 weights=sampling_weights, seed=seed)
    
    valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).cache()
    valid = valid.shuffle(buffer_size=buffer_size, seed=seed).repeat()
    
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).cache()
    test = test.shuffle(buffer_size=buffer_size, seed=seed).repeat()
    
    if onehot:
         train = train.map(partial(one_hot, num_classes=num_classes),
                           num_parallel_calls=AUTOTUNE)
         valid = valid.map(partial(one_hot, num_classes=num_classes),
                           num_parallel_calls=AUTOTUNE)

    train = train.map(standard_scaler, num_parallel_calls=AUTOTUNE)
    valid = valid.map(standard_scaler, num_parallel_calls=AUTOTUNE)
    test = test.map(standard_scaler, num_parallel_calls=AUTOTUNE)

    train = train.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    valid = valid.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test = test.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return train, valid, test, num_classes, steps_per_epoch, validation_steps

def one_hot(img, outputs, num_classes):
    new_outputs = tf.one_hot(outputs, num_classes)
    return img, new_outputs

def labels_to_index_list(text, max_str_len):
    initial_text_len = len(text)
    text = text + " "*(max_str_len-initial_text_len)
    index_list = [labels.index(char) for char in text]
    index_list.append(initial_text_len)
    return np.array(index_list, dtype=object)

if __name__=="__main__":
# =============================================================================
#     train, valid, test, \
#     num_classes, steps_per_epoch, validation_steps = load_emnist(32)
# =============================================================================
    data = DataLoader(2)
    
    gen, steps = data.load_text_data()
    for input, y_true in gen.take(1):
        #print(input)
        #print("---------------------")
        print(y_true)
        y_true = y_true.numpy().astype(np.int32)
        # the labels length is set at the last index of every y_true
        label_length = \
        np.array([i[-1] for i in y_true]).astype(np.int32)
        print(label_length)
        labels = np.zeros(
            (len(label_length), np.max(label_length))).astype(np.int64)
        for nxd, i in enumerate(y_true):
            labels[nxd, :i[-1]] = i[:i[-1]].astype(np.int64)
        print(labels)
