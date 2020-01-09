import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import re
import random

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
        
        self.np_train = np_train
        self.np_test = np_test
        self.index_dict = index_dict
        
    def to_image(self, word, source="emnist"):
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
            word_img.save(f'word_images_emnist/{word}.png')
        elif source == "font":
            word_img = Image.new('L', (len(word)*12+30,28))
            p = Path('fonts')
            all_p = searching_all_files(p)
            index = random.randint(0, len(all_p)) - 1
            fnt = ImageFont.truetype(str(all_p[index]), 15)
            d = ImageDraw.Draw(word_img)
            word_pixel_size = d.textsize(word, font=fnt)
            word_img.resize(word_pixel_size)
            d = ImageDraw.Draw(word_img)
            d.text((word_pixel_size[0]//2,word_pixel_size[1]//3), 
                   word, font=fnt, fill=(255))
            word_img.save(f'word_images/fonts/{word}_.png')

def searching_all_files(directory: Path):   
    file_list = [] # A list for storing files existing in directories

    for x in directory.iterdir():
        if x.is_file():
           file_list.append(x)

    return file_list
        
def to_alphanum(word):
    return re.sub('[^A-Za-z0-9]+', '', word)
        
if __name__=="__main__":
    data = pd.read_csv('words.txt', sep="\n", header=None, 
                       squeeze=True, dtype=str)
    data = data.astype('str') 
    data = data.map(to_alphanum)
    data = data.drop_duplicates()
    data = data.drop([data[data=="Con"].index[0], 
                      data[data=="prn"].index[0]]) # causes error for some reason
    data = data.reset_index(drop = True)
    txt_converter = Text_Converter()
    #data.map(txt_converter.to_image) #generate word images using emnist
    data.map(partial(txt_converter.to_image, source="font")) #generate word images using fonts
    

