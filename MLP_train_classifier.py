import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

'''
This program trains an MLP classifier to predict HXL hashtags.
If you are training a classifier on a new dataset, it is adviced to first tune the parameters of the model.

Input: .csv file containing hashtags, header strings and their corresponding word embeddings from extract_features.py

Output: A pickled MLP classifier. Also the model is tested on a test set, the classification accuracy is printed 
along with the confusion matrix.
'''


def format_embeddings(embedding):
    """Fix some formatting issues from feature extraction"""
    embedding = embedding.replace('\r\n', '')
    embedding = embedding.replace('[', '')
    embedding = embedding.replace(']', '')
    return np.fromstring(embedding, dtype=float, sep=' ').tolist()


input_file = 'wordembedding_data.csv'
output_file = 'MLPclassifier.pkl'

# Read data
df = pd.read_csv(input_file, delimiter=',', encoding='utf-8')

df['Class'] = df['Hashtag']
df['Word_embedding'] = df['Word_embedding'].map(lambda x: format_embeddings(x))

# Remove infrequent classes
threshold = 5   # include only rows with at least this many points
class_count = df['Class'].value_counts()
removal = class_count[class_count <= threshold].index
df['Class'] = df['Class'].replace(removal, np.nan)
df = df.dropna()

df = df[['Class', 'Word_embedding']].copy()
df_labels = df.Class.unique()
df_labels = np.sort(df_labels, axis=-1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Word_embedding'], df['Class'], test_size=0.33, random_state=0)


# Train the classifier with the parameters as specified
clf = MLPClassifier(activation='relu', alpha=0.001, epsilon=1e-08, hidden_layer_sizes=150, solver='adam')
clf.fit(X_train.values.tolist(), y_train.values.tolist())
test_score = clf.score(X_test.tolist(), y_test.tolist())
print("Classification accuracy on test set: %s" %test_score)

# Confusion matrix
y_pred = clf.predict(X_test.values.tolist())
confmatrix = confusion_matrix(y_test.values.tolist(), y_pred, df_labels)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confmatrix, annot=True, fmt='d', xticklabels=df_labels, yticklabels=df_labels, vmax=80)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Save the trained classifier for later use
pickle.dump(clf, open(output_file, 'wb'))
