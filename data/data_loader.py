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

labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class DataLoader():
    
    def __init__(self, batch_size, data_path_1="annotations.csv",
                 data_path_2="y_true.csv",
                 data_path_3="word_images/data/",
                 seed=23, valid_split=.2, test_split=.2):
        annots_raw = pd.read_csv(data_path_1)
        labels_raw = pd.read_csv(data_path_2)
        df_raw = pd.merge(annots_raw, labels_raw, on='index')
        df_raw["index"] = df_raw["index"].map(lambda x: '{}{}.png'.format(data_path_3, x))
        df_raw.rename(columns={"index": "path"}, inplace =True)
        df_raw['path'] = df_raw['path'].astype('|S').str.decode("utf-8")
        df_raw['annotation'] = df_raw['annotation'].astype('|S').str.decode("utf-8")
        df, df_test = train_test_split(df_raw, test_size=test_split, 
                                       random_state=seed)
        df_train, df_valid = train_test_split(df, test_size=valid_split,
                                              random_state=seed)
        df_train = df_train[:int(len(df_train)*.05)]
        df_valid = df_valid[:int(len(df_valid)*.05)]
        df_test = df_test[:int(len(df_test)*.05)]
        self.x_train = df_train["path"]
        self.y_train = df_train['annotation']
        self.label_train = df_train[[str(i) for i in range(31)]]
        self.x_valid = df_valid["path"]
        self.y_valid = df_valid['annotation']
        self.label_valid = df_valid[[str(i) for i in range(31)]]
        self.x_test = df_test["path"]
        self.y_test = df_test['annotation']
        self.label_test = df_test[[str(i) for i in range(31)]]
        self.seed=seed
        self.batch_size=batch_size
        
    def load_text_data(self, type="train"):
        if type=="test":
            paths = self.x_test
            annots = self.y_test
            labels = self.label_test.values
        elif type=="valid":
            paths = self.x_valid
            annots = self.y_valid
            labels = self.label_valid.values
        else:
            paths = self.x_train
            annots = self.y_train
            labels = self.label_train.values
        time_slices = 32
        input_lengths = [time_slices]*len(annots)#[[time_slices]]*len(annots)
        label_lengths = annots.str.len()#.to_numpy().reshape(len(annots), 1)
        steps_per_epoch = np.ceil(len(paths)/self.batch_size)
        input_targets = make_ds(paths,
                                labels,
                                input_lengths,
                                label_lengths,
                                annots, 
                                seed=self.seed)
        imgs_targets = input_targets.map(load_image,
                                         num_parallel_calls=AUTOTUNE)
        imgs_targets = imgs_targets.map(standard_scaler, 
                                        num_parallel_calls=AUTOTUNE)
        ds = imgs_targets.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        return ds, steps_per_epoch
        
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

def standard_scaler(inputs, outputs):
    img = tf.cast(inputs[0], tf.float16)
    img = img/255.0 - .5
    new_inputs = (img, inputs[1], inputs[2], inputs[3])
    return new_inputs, outputs

def make_ds(paths, labels, input_lengths, label_lengths, outputs, seed):
    inputs = (paths, labels, input_lengths, label_lengths)
    ds = tf.data.Dataset.from_tensor_slices((inputs, outputs)).cache()
    ds = ds.shuffle(BUFFER_SIZE, seed=seed).repeat()
    return ds

def load_image(inputs, output, channels=1):
    img = tf.io.read_file(inputs[0])
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.cast(img, tf.uint8)
    new_inputs = (img, inputs[1], inputs[2], inputs[3])
    #tf.print(inputs[1], output_stream=sys.stdout, summarize=31)
    return new_inputs, output

if __name__=="__main__":
# =============================================================================
#     train, valid, test, \
#     num_classes, steps_per_epoch, validation_steps = load_emnist(32)
# =============================================================================
    data = DataLoader(2)
# =============================================================================
#     train_gen = data.load_text_data()
#     valid_gen = data.load_text_data(type="valid")
#     test_gen = data.load_text_data(type="test")
# =============================================================================
