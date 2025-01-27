# This file is meant to run the experiments to determine good keys and values for k.

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pynndescent

import compress
import vectorizer
import dataset
import eval_ir
import neural
import time

"""
Exeperiments for the IR subsystem
"""

# evaluate the IR over a range of different keys for recipe dataset
def run_experiment():
    queries = dataset.get_queries()
    key_list  = [["name"], ["tags", "name"], ["name", "ingredients"], ["tags", "name", "steps"], ["tags", "name", "steps", "ingredients"],["tags", "name", "steps", "ingredients", "description"]]
    #key_list = [["name"], ["tags", "name"]] # change to the keys you want to test
    ngram = False
    for keys in key_list:
        documents, document_ids = dataset.make_documents(keys)
        vector = vectorizer.make_vectorizer(ngram, vectorizer.preprocess)

        start = time.time()
        X = vector.fit_transform(documents)
        fin = time.time()
        print((fin-start)/60)
        
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
        model_knn.fit(X)
        for i in [1, 2, 3, 4, 5, 10, 15, 20, 30]:
            eval_ir.eval_ir(vector, model_knn, keys, queries, document_ids, i, ngram, False, 0) 
        print(f'done fore key={keys}')

# same as above but with 2-grams
def run_experiment_ngram():
    queries = dataset.get_queries()
    key_list  = [["name"], ["tags", "name"], ["name", "ingredients"], ["tags", "name", "steps"], ["tags", "name", "steps", "ingredients"],["tags", "name", "steps", "ingredients", "description"]]
    ngram = True
    for keys in key_list:
        documents, document_ids = dataset.make_documents(keys)
        vector = vectorizer.make_vectorizer(ngram, vectorizer.preprocess)

        start = time.time()
        X = vector.fit_transform(documents)
        fin = time.time()
        print((fin-start)/60)
        
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
        model_knn.fit(X)
        for i in [1, 2, 3, 4, 5, 10, 15, 20, 30]:
            eval_ir.eval_ir(vector, model_knn, keys, queries, document_ids, i, ngram, False, 0) 
        print(f'done fore key={keys}')

#same as above but with TruncatedSVD
def run_experiment_svd():
    queries = dataset.get_queries()
    keys = ["tags", "name", "steps"]
    documents, document_ids = dataset.make_documents(keys)

    vector = vectorizer.make_vectorizer(False, vectorizer.preprocess)

    X = vector.fit_transform(documents)

    for i in [10, 50, 100, 150, 200, 250, 300, 400, 500, 600]:
        n_components = i

        svd = TruncatedSVD(n_components=n_components)
        X_reduced = svd.fit_transform(X)
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
        model_knn.fit(X_reduced)
        for i in [1, 2, 3, 4, 5, 10, 15, 20, 30]:
            eval_ir.eval_ir(vector, model_knn, queries, document_ids, i, False, svd, n_components)
        print(f'done fore n_components={n_components}')

# Experiment with the neural embedder on the paper dataset
def run_experiment_neural():
    queries, anthology_sample = dataset.get_dataset_paper()
    keys = [['title', 'author'], ['title', 'abstract', 'author'], ['title', 'full', 'author']]
    for key in keys:

        documents = dataset.make_documents_paper(keys, anthology_sample)
        preprocesed_docs = [neural.preprocess(doc) for doc in documents]

        #print(preprocesed_docs)
        model = neural.get_neural_embedder()
        document_embeddings = model.encode(preprocesed_docs)
        for k in [1, 2, 3, 4, 5, 10, 15, 20]:

            nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(document_embeddings)
            eval_ir.eval_ir_neural(nbrs, k, key, queries, model, anthology_sample)

        print(f'Done with {key}')

# experiments using word2Vec
def run_experiment_word_embed():

    queries, anthology_sample = dataset.get_dataset_paper()
    keys = [['title', 'author'], ['title', 'abstract', 'author'], ['title', 'full', 'author']]
    for key in keys:

        documents = dataset.make_documents_paper(keys, anthology_sample)
        preprocesed_docs = [neural.preprocess(doc) for doc in documents]
        w2v_model, vector_size = neural.get_neural_word_embded(preprocesed_docs)

    document_vectors = np.zeros((len(preprocesed_docs), vector_size))
    
    for i, doc in enumerate(preprocesed_docs):
        document_vectors[i, :] = neural.get_embedding_w2v(w2v_model, doc, vector_size)

    for k in [1, 2, 3, 4, 5, 10, 15, 20]:    
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(document_vectors)
        eval_ir.eval_ir_word(nbrs, k, key, queries, vector_size ,anthology_sample)
    print(f'\n Done with {key}')

# experiments with faster NN (NNDescent)
def run_experiment_fast():

    queries, anthology_sample = dataset.get_dataset_paper()
    keys = [['title', 'author'], ['title', 'abstract', 'author'], ['title', 'full', 'author']]

    for key in keys:

        documents = dataset.make_documents_paper(keys, anthology_sample)
        preprocesed_docs = [neural.preprocess(doc) for doc in documents]

        model = neural.get_neural_embedder()
        document_embeddings = model.encode(preprocesed_docs)
        index = pynndescent.NNDescent(document_embeddings, n_neighbors=400, metric='cosine', n_trees=100, n_iters=80, max_candidates=400, low_memory=True)

        for k in [1, 1, 2, 3, 4, 5, 10, 15, 20]:    
            eval_ir.eval_ir_fast(index, k, key, queries, model, anthology_sample)
        print(f'Done with {key}')

#Experiments with compressed documents
def run_experiment_comp():
    key = ['sum']

    queries, anthology_sample = dataset.get_dataset_paper()
    documents = compress.get_summaries_bert()

    preprocesed_docs = [neural.preprocess(doc) for doc in documents]

    model = neural.get_neural_embedder()
    document_embeddings = model.encode(preprocesed_docs)
    for k in [1, 2, 3, 4, 5, 10, 15, 20]:    
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(document_embeddings)
        eval_ir.eval_ir_sum(nbrs, k, queries, key, model, anthology_sample)
    print(f'Done')
