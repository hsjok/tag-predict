import pandas as pd
import matplotlib.pyplot as plt
import wordcloud
from os import path

'''
This program generates wordclouds of the table headers associated with different HXL hashtags.

Input: .csv file containing at least the columns 'Hashtag' and 'Text header'

Output: A set of .png figures of the word clouds for the hashtags
'''


input_file = "hdx-hashtags-list.csv"    # This is the raw HXL csv I got from David Megginson


d = path.dirname(__file__)

# Read and process data
df = pd.read_csv(input_file)
df.columns = df.columns.str.lower()
cols = df.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, bytes)) else x)
df.columns = cols
df["text_header"] = df["text_header"].str.lower()
df["text_header"] = df["text_header"].replace('_', ' ', regex=True)

tagList = df.hashtag.unique()   # List all unique hashtags in the dataset
output = pd.DataFrame(columns=['Hashtag','Count','Unique headers','Score'])

i=0
for tag in tagList:
    # Compute various statistics
    df_tag = df.loc[df['hashtag'] == tag]
    count = df_tag.shape[0]
    unique = len(df_tag['text_header'].unique())
    output.loc[i] = [tag, count, unique, unique/count]
    i+=1

output_top100 = output.loc[output["Count"]>100]  # Create word clouds only for the tags with >100 occurrences

for index, row in output_top100.iterrows():
    # Create wordclouds
    hashtag = row['Hashtag']
    df_wc = df.loc[df['hashtag'] == hashtag]
    tuples = tuple([tuple(x) for x in df_wc.text_header.value_counts().reset_index().values])
    tuples = dict(tuples)
    cloud = wordcloud.WordCloud(background_color="white", max_font_size=40)
    cloud.generate_from_frequencies(tuples)
    plt.figure()
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.text(70,230,"Hashtag Occurrence: %s, Unique Headers: %s" %(row['Count'], row['Unique headers']))
    plt.title(hashtag, fontsize=18)
    plt.savefig(path.join(d, "wordcloud", "wordcloud%s.png" %hashtag))
    plt.close()

