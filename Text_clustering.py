from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import string

# download stopwords dictionary for running for the first time
'''
nltk.download('stopwords')
nltk.download('punkt')
'''
import matplotlib.pyplot as plt
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from gensim import corpora, models, similarities
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import ward, dendrogram
from nltk.tag import pos_tag

# Lists involves:
# 1. 'titles': the titles of the films in their rank order mostly used for labeling
# 2. 'synopses': the synopses of the films matched to the 'titles' order from both Wiki and IMDB
# 3. 'links' for imdb data
# 4. 'genres' list of movies

# Titles list
titles = open('data/title_list.txt').read().split('\n')
# From all only select first 100
titles = titles[:100]

# Links list
links = open('data/link_list_imdb.txt').read().split('\n')
links = links[:100]

# Synopses from Wiki list
synopses_wiki = open('data/synopses_list_wiki.txt').read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]

synopses_clean_wiki = []
for text in synopses_wiki:
    # strips html formatting and converts to unicode
    text = BeautifulSoup(text, 'html.parser').getText()
    synopses_clean_wiki.append(text)

synopses_wiki = synopses_clean_wiki

# Synopses from IMDB list
synopses_imdb = open('data/synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]

synopses_clean_imdb = []

for text in synopses_imdb:
    # strips html formatting and converts to unicode
    text = BeautifulSoup(text, 'html.parser').getText()
    synopses_clean_imdb.append(text)

synopses_imdb = synopses_clean_imdb

# Genres list
genres = open('data/genres_list.txt').read().split('\n')
genres = genres[:100]
'''
print(str(len(titles)) + ' titles')
print(str(len(links)) + ' links')
print(str(len(synopses_wiki)) + ' wiki_synopses')
print(str(len(synopses_imdb)) + ' imdb_synopses')
print(str(len(genres)) + ' genres')
'''
# For sneak peak, first 10 titles can be seen
# print(titles[:10])

# Make combined synopses list
synopses = []
for i in range(len(synopses_wiki)):
    item = synopses_wiki[i] + synopses_imdb[i]
    synopses.append(item)

# generates index list for each item in the text corpus and later can be used for scoring
ranks = []
for i in range(0, len(titles)):
    ranks.append(i)

# Manipulating the synopses
# 1. Defining objects
# Removal of "Stp words" in english
stopwords = nltk.corpus.stopwords.words('english')

# Stemming or breaking down words into its root
stemmer = SnowballStemmer("english")


# 2. Defining functions for data processing and tokenizing (stop words and stemming)
# splits the synopsis into a list of its respective words or tokenize corpus
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# splits the synopsis into a list of its respective words or tokenize them and also stems each token
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# Create a list with stemmed+tokenizing and tokenizing only from synapses list by using above defined two functions
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

# Creating a data frame with stemmed vocbulary as the index and tokernized words list as the column.
# This procides an efficent way to look up a stem (word) and return a full token for further processing
# However stem tokers are one to many such as stem 'run' could came from 'running', 'ran' or 'runs'
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

'''
TFIDF and document similarity
Define term frequency-inverse document frequency (tf-idf) vectorizer object (parameters) the convert the synopses list 
into a tf-idf matrix using the vectorizer object
Tasks involved:
1. Develop document term frequency measurement by counting the word frequency occurrences.
2. Get term frequency-inverse document frequency (TF-IDF) matrix to identify how frequently occur or how important words 
   to a documents (or collection of corpus)
3. This contains TF-IDF weights measured based on the importance of words and higher weight add more meaning to the doc
'''
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

# print(tfidf_matrix.shape)  # inspections

terms = tfidf_vectorizer.get_feature_names()

# Appy cosine similartity to get similarity distance between features of TF-IDF matrix
dist = 1 - cosine_similarity(tfidf_matrix)

# tf-idf matrix, you can run a slew of clustering algorithms to better understand the hidden structure within  synopses.
# Run K-means clustering to get a better understanding of hidden structure with in the synopses using TF-IDF matrix
num_clusters = 5
km = KMeans(n_clusters=num_clusters)  # Clustering object
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()  # # Labeling clusters

# Data framing the results
films = {'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters, 'genre': genres}

frame = pd.DataFrame(films, index=[clusters], columns=['rank', 'title', 'cluster', 'genre'])
grouped = frame['rank'].groupby(frame['cluster'])

# Write clustered data to a CSV file
frame.to_csv('clustered_movies.csv')

print("Top terms per cluster:")
print("\n")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster no %d words (common):" % (i + 1), end='')
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print("\n")
    print("Cluster no %d titles:" % (i + 1), end='')
    for title in frame.loc[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print("\n")

# Hierarchical document clustering approach
# define the linkage_matrix using ward clustering pre-computed distances
linkage_matrix = ward(dist)

fig, ax = plt.subplots(figsize=(15, 20))  # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles)

plt.tick_params(axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom='off',  # ticks along the bottom edge are off
                top='off',  # ticks along the top edge are off
                labelbottom='off')

plt.tight_layout()
plt.savefig('ward_clusters.png', dpi=200)
plt.close()


# Latent Dirichlet Allocation ----------
# 1. Strip any proper names from a text
def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


# 2. strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text
def strip_proppers_POS(text):
    tagged = pos_tag(text.split())  # use NLTK's part of speech tagger
    non_propernouns = [word for word, pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns


# Latent Dirichlet Allocation implementation
# remove proper names
preprocess = [strip_proppers(doc) for doc in synopses]
tokenized_text = [tokenize_and_stem(text) for text in preprocess]
texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=1, no_above=0.8)
corpus = [dictionary.doc2bow(text) for text in texts]

# LDA model fit
lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary, update_every=5, chunksize=10000, passes=100)
topics = lda.print_topics(5, num_words=20)
topics_matrix = lda.show_topics(formatted=False, num_words=20)
topics_matrix = np.array(topics_matrix)
print(topics_matrix.shape)

topic_words = topics_matrix
for i in topic_words:
    print([str(word) for word in i])
    print("\n")
