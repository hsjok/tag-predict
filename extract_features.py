from fastText import load_model
import pandas as pd
import re


'''
This program extracts features from the cleaned HXL headers by converting them to word embeddings.
The word embeddings used here are 300-dim fastText embeddings. They are loaded from a large (~10GB) fastText model
which can be downloaded here: https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
Note that the .zip contains a .bin file and a .vec file. These are different formats for storing the fastText model,
use the .bin file whenever possible.

Input: .csv file where each row contains a hashtag and a cleaned header string

Output: .csv file containing hashtags, header strings and their corresponding word embeddings
NOTE: this output formatting is not ideal and currently has to be handled ad hoc in the program which trains the ML 
model. It should be changed to something more suitable for storing large vectors, e.g. .xml, .pickle, etc.
'''


input_file = 'cleaned_hxl_data.csv'
output_file = 'wordembedding_data.csv'
pretrained_fasttext_model = 'wiki.en.bin'   # https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip


# Load the fastText model
fastText_model = load_model(pretrained_fasttext_model)
print("Pre-trained model loaded successfully!\n")

# Read the cleaned HXL data
df = pd.read_csv(input_file , delimiter=',', encoding='utf-8')
df["Text_header"] = df["Text_header"].map(lambda x: re.sub(' +', ' ', str(x)))

# Get a vector representation of each header
df['Word_embedding'] = df['Text_header'].map(lambda x: fastText_model.get_sentence_vector(str(x)))
print("Word embeddings extracted!\n")

# Save the vectorized data
df.to_csv(output_file, sep=',', encoding='utf-8', index=False)