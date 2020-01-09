import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_emnist

def plot_dist(data, y_label, x_label):
    data.head(62).plot.bar(figsize=(15,10))
    plt.xticks(rotation=360);
    plt.xlabel(x_label);
    plt.ylabel(y_label);    
    plt.show()

if __name__=="__main__":
    batch_size = 128
    train, valid, test, num_classes, \
    steps_per_epoch, validation_steps = load_emnist(batch_size, onehot=False)
    
    label_indeces = []
    for inputs, outputs in train.take(steps_per_epoch):
        label_indeces.extend(outputs.numpy())
    label_indeces = pd.Series(label_indeces)
    
    numbers = "0123456789"
    letters_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letters_lower = "abcdefghijklmnopqrstuvwxyz"
        
    classes = list(numbers + letters_upper + letters_lower)
    index_dict = {v: k for v, k in enumerate(classes)}
    
    labels = label_indeces.map(index_dict)
    
    freq_labels = labels.value_counts().sort_index()
    plot_dist(freq_labels, "Count", "Character")
