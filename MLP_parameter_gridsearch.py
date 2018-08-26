import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pickle

'''
This program tunes the parameters of the Multilayer Perceptron model through a crossvalidated gridsearch.

Input: .csv file containing hashtags, header strings and their corresponding word embeddings from extract_features.py

Output: A pickled MLP classifier. Also the optimal parameter values are printed.
'''


def format_embeddings(embedding):
    """Fix some formatting issues from feature extraction"""
    embedding = embedding.replace('\r\n', '')
    embedding = embedding.replace('[', '')
    embedding = embedding.replace(']', '')
    return np.fromstring(embedding, dtype=float, sep=' ').tolist()


input_file = 'wordembedding_data.csv'
output_file = 'MLPclassifier.pkl'


# Read and process data
df = pd.read_csv(input_file, delimiter=',', encoding='utf-8')

df['Class'] = df['Hashtag']
df['Word_embedding'] = df['Word_embedding'].map(lambda x: format_embeddings(x))

threshold = 5   # include only rows with at least this many points
class_count = df['Class'].value_counts()
removal = class_count[class_count <= threshold].index
df['Class'] = df['Class'].replace(removal, np.nan)
df = df.dropna()

df = df[['Class', 'Word_embedding']].copy()

X = df['Word_embedding'].values.tolist()
y = df['Class'].values.tolist()


# Parameter grid to search through
param_grid = [
    {
        'solver' : ['adam', 'lbfgs'],
        'alpha' : [0.001, 0.01, 0.1],
        'hidden_layer_sizes' : [50, 75, 100, 150, 200],
        'activation' : ['tanh', 'relu']
    }
]

# Tune parameters
clf = GridSearchCV(MLPClassifier(), param_grid, cv=3, scoring='accuracy', verbose=10)
clf.fit(X,y)
print("Best parameters set found on development set:")
print(clf.best_params_)

# Save trained classifier
pickle.dump(clf, open(output_file, 'wb'))


# 2018-08-20: Good parameter choices found to be:
# 'activation': 'relu', 'alpha': 0.001, 'epsilon': 1e-08, 'hidden_layer_sizes': 150, 'solver': 'adam'
