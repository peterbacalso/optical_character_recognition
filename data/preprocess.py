from PIL import Image
from pathlib import Path
import csv
import pandas as pd
import numpy as np
import time
# 910661

all_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "

def resize_and_pad(file, save_dir=None):
    old_im = Image.open(file)
    old_size = old_im.size

    if (old_size[0] > 128): 
        old_im.thumbnail((128, 32), Image.ANTIALIAS)
        old_size = old_im.size
    
    new_size = (128, 32)
    new_im = Image.new("L", new_size)
    new_im.paste(old_im, ((new_size[0]-old_size[0])//2,
                          (new_size[1]-old_size[1])//2))
    
    if save_dir is not None:
        file_name = file.name
        new_im.save(f'{save_dir}/{file_name}')
    else:
        new_im.show()
        
def replace_and_annotate(img_dir, start_index=0):
    annotations = []
    index = start_index
    for f in img_dir:
        split_path = str(f).split('\\')
        f.replace("\\".join(split_path[:-1]) + f"\\{index}.png")
        str_value = split_path[-1].split('.')[0]
        annotations.append([index, str_value[:-1]])
        index += 1
    with open('annotations.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(annotations)
        
        
def generate_text_to_labels():
    df_raw = pd.read_csv("annotations.csv")
    df_raw['annotation'] = df_raw['annotation'].astype('|S').str.decode("utf-8")
    labels = []
    for i in range(len(df_raw)):
        label_row = convert_to_ascii(df_raw['annotation'][i])
        label_row = np.insert(label_row, 0, df_raw['index'][i])
        labels.append(label_row)
    with open('y_true.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(labels)
    
def convert_to_ascii(text):
    text = text + " "*(31-len(text))
    return np.fromiter(map(chart_to_label, text), int, count=31)

def chart_to_label(char):
    return all_labels.index(char)
    


if __name__=="__main__":
# =============================================================================
#     font_dir = Path('word_images/fonts').glob('**/*')
#     emnist_dir = Path('word_images/emnist').glob('**/*')
#     
#     #replace_and_annotate(font_dir)
#     #replace_and_annotate(emnist_dir, 455330)
#     
#     save_dir = 'word_images/data/'
#     [resize_and_pad(f, save_dir) for f in font_dir]
#     [resize_and_pad(f, save_dir) for f in emnist_dir]
# =============================================================================
    generate_text_to_labels()
