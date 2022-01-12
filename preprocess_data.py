import os
import re
import glob
import pandas as pd
import pickle
import argparse
import numpy as np
import json
import nltk
from gensim import corpora
from nltk.corpus import stopwords
from torch import nn
from sentence_transformers import SentenceTransformer, models

nltk.download('stopwords')
stops_fi = set(stopwords.words('finnish'))

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='yle/ylenews-fi-2011-2018-src/data/fi', help='Folder containing data')
parser.add_argument('--save_path', type=str, default='data//', help='Path to folder where results are saved')

args = parser.parse_args()

# Reads the .josn files from the given folder, extracts relevant fields and saves the articles in 
# a Pandas dataframe to a given location
def read_corpus(folder):
    print(folder)
    files = [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.json'))]

    articles = []

    for filepath in files:

        with open(filepath, 'r') as f:
            try:
                text = json.load(f)
            except:
                print('errors', filepath)
                continue
            
            for a in text['data']:
                article = {'id': a['id'], 'date': a['datePublished']}
                body = ""
                title = ""
                subjects = []
                
                for c in a['content']:
                    if c['type'] == 'heading':
                        title = c['text']
                    elif c['type'] == 'text':
                        body += " " + c['text']
                
                for s in a.get('subjects', []):
                    if s['title']['fi']:
                        subjects.append(s['title']['fi'].lower())
                    
                article['title'] = title
                article['body'] = body
                article['subjects'] = subjects
                
                articles.append(article)
        
    return articles

def preprocess(dataframe):

    df = dataframe.copy()

    # Join article titles and text body
    df['body'] = df['title'] + ' ' + df['body']

    # Drop title column
    df = df.drop(['title'], axis=1)

    df['body'].replace(['',' '], np.nan, inplace=True)
    df.dropna(subset=['body'], inplace=True)

    #Tokenize documents
    data_tokens = [[text for text in doc.split()] for doc in df['body']]

    #Filter out stopwords, short tokens and non-letter tokens
    tokens = [[token.lower() for token in doc if token.lower() not in stops_fi and len(token) > 2] for doc in data_tokens]

    #Form a Gensim dictionary from the tokenized documents
    doc_dict = corpora.Dictionary(tokens)

    #Filter out very rare and common tokens 
    doc_dict.filter_extremes(no_below=5, no_above=0.7)

    new_docs = []
    for i, doc in enumerate(tokens):
        new_doc = " ".join([t for t in doc if doc_dict.doc2bow([t])])
        new_docs.append(new_doc)
                            
    df['body'] = new_docs

    return df

def get_embeddings(df):

    # Extract article bodies from df
    texts = list(df['body'])
    
    #Create the embedding model
    embedding_pooling_model = SentenceTransformer('all-distilroberta-v1')

    embedding_pooling_model.max_seq_length = 512

    #fully connected dense layer with Tanh activation, which performs a down-projection to 128 dimensions
    dense_model = models.Dense(in_features=embedding_pooling_model.get_sentence_embedding_dimension(), out_features=300, activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[embedding_pooling_model,dense_model])

    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Get ids of the articles from df
    ids = list(df['id'])

    # Crete a dict with article embeddings and corresponding ids
    res = dict(zip(ids, embeddings))

    return res


# Extract data as a list of dicts
data = read_corpus(args.data_path)

# Get data in dataframe form
df = pd.DataFrame(data)

# Save dataframe in pickle form
df.to_pickle(args.save_path + 'yle_df.pkl')

# Preprocess data
filtered_df = preprocess(df)

# Get article embeddings in dict form
embeddings_dict = get_embeddings(filtered_df)

# Extract 'id' and 'subjects' columns from dataframe
df_light = filtered_df[['id', 'subjects']].copy()

# Save article embeddings
with open(args.save_path + 'embeddings_dict.pkl', 'wb') as handle:
    pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the dataframe with 'id' and 'subjects' columns
with open(args.save_path + 'df_light.pkl', 'wb') as handle:
    pickle.dump(df_light, handle, protocol=pickle.HIGHEST_PROTOCOL)