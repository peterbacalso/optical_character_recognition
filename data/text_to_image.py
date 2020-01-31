import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import re
import random
import time
import csv

from functools import partial
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

class Text_Converter():
    
    def __init__(self):
        dataset = tfds.load(name="emnist", batch_size=32, 
                            as_supervised=True, shuffle_files=True)
        train = dataset['train'].repeat()
        test = dataset['test'].repeat()
        np_train = tfds.as_numpy(train)
        np_test = tfds.as_numpy(test)

        numbers = "0123456789"
        letters_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letters_lower = "abcdefghijklmnopqrstuvwxyz"
        special = " "
        
        classes = list(numbers + letters_upper + letters_lower + special)
        index_dict = {k: v for v, k in enumerate(classes)}
        
        self.classes = classes
        self.np_train = np_train
        self.np_test = np_test
        self.index_dict = index_dict
        
    def to_image(self, word, save_dir, source="emnist", img_type='jpg'):
        if source == "emnist":
            letter_index_list = [self.index_dict[char] for char in word]
            letter_img_list = []
            curr_index = 0
    
            for item in self.np_train:
                if (letter_index_list[curr_index] == 62):
                    letter_img_list.append(np.zeros((28,28)))
                    curr_index += 1
                    continue
                if (letter_index_list[curr_index] in item[1]):
                    matching_index = np.where(item[1]==letter_index_list[curr_index])
                    letter_img_list.append(np.reshape(item[0][matching_index][-1], 
                                                      (28,28)).T)
                    curr_index += 1
                if (curr_index == len(word)):
                    break
    
            word_array = np.concatenate(letter_img_list,axis=1)
            word_img = Image.fromarray(word_array)
            word_img = word_img.convert("L")
            word_img.save(f'{save_dir}/{word}.{img_type}')
        elif source == "font":
            W, H = (32,32)
            word_img = Image.new('L', (W,H))
            p = Path('fonts')
            all_p = searching_all_files(p)
            font_index = random.randint(0, len(all_p)) - 1
            fnt = ImageFont.truetype(str(all_p[font_index]), 15)
            draw = ImageDraw.Draw(word_img)
            w, h = draw.textsize(word, font=fnt)
            draw.text(((W-w)/2,(H-h)/2), 
                      word, font=fnt, fill=(255))
            #word_img.save(f'{save_dir}/{word}.{img_type}')
            word_matrix = np.array(word_img)
            word_matrix = word_matrix.flatten()

            row = np.concatenate((np.array([word]), word_matrix))
            annotations = [row]
            with open('char_font_dataset.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(annotations)

def searching_all_files(directory: Path):   
    file_list = [] # A list for storing files existing in directories

    for x in directory.iterdir():
        if x.is_file():
           file_list.append(x)

    return file_list
        
def to_alphanum(word):
    return re.sub('[^A-Za-z0-9]+', '', word)

def index_to_label(index, label_list):
    return label_list[index]
        
if __name__=="__main__":
    
# =============================================================================
#     # words dataset
#     data = pd.read_csv('words.txt', sep="\n", header=None, 
#                        squeeze=True, dtype=str)
#     data = data.astype('str') 
#         data = data.map(to_alphanum)
#     data = data.drop_duplicates()
#     data = data.drop([data[data=="Con"].index[0], 
#                       data[data=="prn"].index[0]]) # causes error for some reason
#     data = data.reset_index(drop = True)
#     #data.map(partial(txt_converter.to_image, save_dir='word_images_emnist')) #generate word images using emnist
#     data.map(partial(txt_converter.to_image, 
#                      save_dir='word_images/fonts', 
#                      source="font")) #generate word images using fonts
# =============================================================================
    
    start = time.time()
    # char dataset
    char_data_indices = np.random.randint(62, size=1)
    txt_converter = Text_Converter()
    to_label = np.vectorize(partial(index_to_label, 
                                    label_list=txt_converter.classes))
    char_data = to_label(char_data_indices)
    char_series = pd.Series(char_data)
    #char_df = char_series.reset_index()
    char_series.map(partial(txt_converter.to_image, 
                            save_dir='characters/font', 
                            source="font"))
# =============================================================================
#     char_df.apply(partial(txt_converter.to_image, 
#                           save_dir='characters/font', 
#                           source="font"), axis=1)
# =============================================================================
    end = time.time()
    print(end - start)
    

