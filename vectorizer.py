from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
import pandas as pd
import numpy as np
import pickle
import re
import time

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(doc):
    words = word_tokenize(re.sub(r'[-]', ' ', doc))
    word_tags = pos_tag(words)
    
    preprocessed_text = [
        lemmatizer.lemmatize(word.lower())
        for word, pos in word_tags
        if (pos.startswith('NN') or pos.startswith('VB')) and word.lower() not in stopwords
    ]
    
    preprocessed_text = [
        word.translate(str.maketrans('', '', string.punctuation))
        for word in preprocessed_text
        if word
    ]
    
    return preprocessed_text


def make_vectorizer(ngram, preprocess):   
    if ngram:
        vectorizer = TfidfVectorizer(min_df=1, 
                                    max_df = 0.9, 
                                    sublinear_tf=True, 
                                    use_idf =True, 
                                    smooth_idf=True,
                                    ngram_range=(1, 2),
                                    norm="l2",
                                    tokenizer=preprocess) 
    else :
        vectorizer = TfidfVectorizer(min_df=1, 
                                max_df = 0.9, 
                                sublinear_tf=True, 
                                use_idf =True, 
                                smooth_idf=True,
                                #ngram_range=(1, 2),
                                norm="l2",
                                tokenizer=preprocess) 
    return vectorizer

def fit_vectorizer(vectorizer, documents):
    start = time.time()
    X = vectorizer.fit_transform(documents)
    fin = time.time()
    print((fin-start)/60)