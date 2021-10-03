#!/usr/bin/env python
# coding: utf-8

"""
    This programms scrapes lyrics  and trains a Machine Learning Model
    to predict the artist of the songs.
"""

import nltk
import requests
import re
import time
import os
import sys
from bs4 import BeautifulSoup
import pandas as pd
from nltk import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from nltk.corpus import stopwords
import pickle
nltk.download('stopwords')
nltk.download("wordnet")



artists = [artist_1, artist_2]

def get_links(artists):
    """
    get all sublinks to artists' songs
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36'}
    timer = 5
    list_artist_1 = []
    list_artist_2 = []
    count = 0
    for person in artists:
        time.sleep(5)
        response = requests.get(person, headers)
        response, type(response) 
        all_links = re.findall('href=\"(/lyric.*?)\"', response.text) 
        if count == 0:
            list_artist_1.append(all_links[:50])
        else:
            list_artist_2.append(all_links[:50])
        count += 1
    list_artist_1 = list_artist_1[0]
    list_artist_2 = llist_artist_2[0]
    new_link = list_artist_1 + list_artist_2
    return new_link




def get_complete_links(new_link):
    """ 
    returns complete links to lyrics
     """
    start_url = 'https://www.lyrics.com'
    complete_links = []
    for x in new_link:
        combine = start_url + x
        complete_links.append(combine)
    return complete_links



def get_text_to_list(complete_links):
    """
    parse pages with beautiful soup and extract lyrics. Write lyrics as string to list
    """
    lyrics = [] 
    for link in complete_links:
        time.sleep(5)
        response = requests.get(link)
        songs = response.text
        songs_soup = BeautifulSoup(songs, 'html.parser')
        filler = [t.text for t in songs_soup.find_all('pre', attrs={'id': 'lyric-body-text'})] 
        lyrics.append(filler[0])
    return lyrics



def get_clean_corpus(CORPUS):

    """
    clean and preprocess text corpus for Machine Learning. Add Tokenizer and Lemmatizer.
    """
    CORPUS = lyrics
    CORPUS = [s.lower() for s in CORPUS]
    CLEAN_CORPUS = []
    tokenizer = TreebankWordTokenizer()
    lemmatizer = WordNetLemmatizer()
    for doc in CORPUS:
        tokens = tokenizer.tokenize(text=doc)
        clean_doc = " ".join(lemmatizer.lemmatize(token) for token in tokens)
        CLEAN_CORPUS.append(clean_doc)
    return CLEAN_CORPUS


def get_labels(artist_1, artist_2):
    """
    Add artists' labels.
    """
    LABELS = [artist_1] * 50 + [artist_2] * 50
    return LABELS



def train_model_1(CLEAN_CORPUS, LABELS):
    """
    Build a pipeline model.
    """
    STOPWORDS = stopwords.words('english')
    pipe = make_pipeline(CountVectorizer(stop_words=STOPWORDS),LogisticRegression())
    model = pipe.fit(CLEAN_CORPUS, LABELS)
    return model


def train_model2(clean_corpus, labels):
    """
    Train a second model, add TfidfVectorizer
    """
    STOPWORDS = stopwords.words('english')
    steps = [('tf-idf', TfidfVectorizer(stop_words=STOPWORDS)), ('LR', LogisticRegression())]
    pipeline = Pipeline(steps)
    model = pipeline.fit(clean_corpus, labels)
    return model



if __name__ == '__main__':
    """
    Saves models as pickle file in current working directory.
    """
    artist_1 = sys.argv[1]
    artist_2 = sys.argv[2]

    new_link = get_links(artists)
    complete_links = get_complete_links(new_link)
    lyrics = get_text_to_list(complete_links)
    CLEAN_CORPUS = get_clean_corpus(lyrics)
    LABELS = get_labels(artist_1, artist_2)
  
    model = train_model1(clean_corpus, labels)
    pickle.dump(model, open('lyrics.model1', 'wb'))

    model = train_model2(clean_corpus, labels)
    pickle.dump(model, open('lyrics.model2', 'wb'))

