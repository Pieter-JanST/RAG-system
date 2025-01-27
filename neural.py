from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
import multiprocessing
import gensim
import string
import re
import vectorizer
import numpy as np

"""
Functions for neural embeddings
"""

#Preprocess function used for the paper dataset
def preprocess(doc):
    stopword = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(re.sub(r'[-]', ' ', doc))

    preprocessed_text = [
        lemmatizer.lemmatize(word.lower().translate(str.maketrans('', '', string.punctuation)))
        for word in words 
        if word and word not in stopword
    ]
    
    return ' '.join(preprocessed_text)

# Get the neural embedder models
def get_neural_embedder():
    model_name = 'multi-qa-mpnet-base-cos-v1'
    model = SentenceTransformer(model_name, device="cuda")
    #model.save(model_name)
    return model

# Build and return a W2V model trained on the given documents
def get_neural_word_embded(documents):
    window = 5
    vector_size = 2000
    epoch=30
    w2v_model = Word2Vec(min_count=1,
                         window=window,
                         vector_size=vector_size,
                         workers=5)
    
    preprocesed_docs = [preprocess(doc) for doc in documents]
    
    vocab = []
    for doc in preprocesed_docs:
        vocab.append(doc.split())

    w2v_model.build_vocab(vocab, progress_per=1000)
    w2v_model.train(preprocesed_docs, total_examples=w2v_model.corpus_count, epochs=epoch, report_delay=1)
    
    # L2 Reg
    w2v_model.wv.vectors /= np.linalg.norm(w2v_model.wv.vectors, axis=1)[:, np.newaxis]

    return w2v_model, vector_size

# Can be changed from average pooling to sum pooling, similar results
def get_embedding_w2v(w2v_model, doc_tokens, vector_size):
    embeddings = []
    # Split doc into words
    doc_tokens = doc_tokens.split()
    if len(doc_tokens)<1:
        return np.zeros(vector_size)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv.key_to_index:
                embeddings.append(w2v_model.wv.get_vector(tok))
        if embeddings:
            return np.mean(embeddings, axis=0) # mean pooling
            #return np.sum(embeddings, axis=0) # max pooling
        else:
            return np.zeros(vector_size)