import pandas as pd
from fastText import load_model
from os import path
import re, string
import pickle

'''
This program reads an untagged dataset and tags it using a trained classifier.

Input: Raw .xlsx file without tags from the HDX.
NOTE: This PoC has been written for .xlsx files but could easily be rewritten to handle other formats

Output: The same .xlsx but with an additional row containing the predicted hashtags
'''


def split_punctuation(value): # split strings on punctuation characters:
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return value.translate(table)


def lower_case_cond(value): # lowercase only words which are all uppercase
    word_list = value.split()
    for i, word in enumerate(word_list):
        if word.isupper():
            word_list[i] = word.lower()
    return ' '.join(word_list)


def split_uppercase(value): # split strings on uppercase
    return re.sub(r'([A-Z])', r' \1', str(value))


def remove_excess_whitespace(value):
    return ' '.join(value.split())


def format_header(header):
    header = str(header)
    header = split_punctuation(header)
    header = lower_case_cond(header)
    header = split_uppercase(header)
    header = remove_excess_whitespace(header)
    header = header.lower()
    return header


input_file = "data.xlsx"
pretrained_fasttext_model = 'wiki.en.bin'   # https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip

d = path.dirname(__file__)
df = pd.read_excel(path.join(d, "..", "Unlabeled Test Data", input_file))    # Path to untagged dataset

# Preprocessing
headers = list(df)
headers = [format_header(x) for x in headers]

# Load word embedding model for feature generation
fastText_model = load_model(pretrained_fasttext_model)
print("Pre-trained model loaded successfully!\n")

# Convert dataset headers into word embeddings
headers = [fastText_model.get_sentence_vector(x).tolist() for x in headers]

# Load the pre-trained classifier
clf = pickle.load(open("MLPclassifier.pkl", 'rb'))

# Predict tags
tags = clf.predict(headers)

# Insert row of tags into the dataset
df.loc[-1] = tags
df.index = df.index + 1  # shifting index
df.sort_index(inplace=True)

writer = pd.ExcelWriter(path.join(d,"..","Unlabeled Test Data","Tagged-"+input_file), engine='xlsxwriter')
df.to_excel(writer, index=False)
writer.save()


