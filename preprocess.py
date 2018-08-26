import pandas as pd
from os import path
import re, string


'''
This program pre-processes the data consisting of the headers of HDX datasets.
The idea is to prepare the strings for the word embedding model which works best if words are separated by 
blank space. Thus the main pre-processing steps are to split the strings on punctuation characters, split on
single capital letters, lowercase everything and remove excess whitespace.

Input: .xlsx file containing at least the columns 'Hashtag' and 'Text header'. It is recommended that the input file
be deduplicated so as not to include repetitions of identical file structures.

Output: .csv file where each row contains a hashtag and a cleaned header string
'''


def split_uppercase(value):     # split strings on uppercase
    return re.sub(r'([A-Z])', r' \1', str(value))


def lower_case_cond(value):     # lowercase only words which are all uppercase
    word_list = value.split()
    for i, word in enumerate(word_list):
        if word.isupper():
            word_list[i] = word.lower()
    return ' '.join(word_list)


def split_punctuation(value):   # split strings on punctuation characters:
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return value.translate(table)


def remove_excess_whitespace(value):
    return ' '.join(value.split())


input_file = 'hxl-hashtags-and-headers-DEDUPLICATED-20180807.xlsx'
output_file = 'cleaned_hxl_data.csv'


d = path.dirname(__file__)

df = pd.read_excel(input_file)
label = df[['Hashtag']]

df['Text header'] = df['Text header'].map(lambda x: str(x))
df['Text header'] = df['Text header'].map(lambda x: split_punctuation(x))
df['Text header'] = df['Text header'].map(lambda x: lower_case_cond(x))
df['Text header'] = df['Text header'].map(lambda x: split_uppercase(x))
df['Text header'] = df['Text header'].map(lambda x: remove_excess_whitespace(x))
df['Text header'] = df['Text header'].map(lambda x: x.lower())

header = df[['Text header']]

training_data = pd.concat([label, header], axis=1)
training_data.to_csv(path.join(d, output_file), index=False, sep=',', encoding='utf-8', quotechar=' ')
