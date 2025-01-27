import torch
import transformers
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import vectorizer
import dataset
import neural
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
Run the LLM model for the given query and documents
"""

token = 'provide valid token'

# load the model and the tokenizer
def get_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    return model_id, model, tokenizer

# run the RAG system over all the queries
def run_rag_all(prompt, model, tokenizer, documents):
    if not prompt:
        prompt=f"""
        [INST]<s>Your job is to answer the given query by leveraging the knowledge in the provided documents. These documents might be relevant, but may also be irrelevant. Each document will have a distance listed with it, this indicates how relevant the document is. Try to use the documents with the lowest distance first. Only use information from relevant documents. Combine the information found in the different relevant documents and produce a compact, short and comprehensive answer to the query. In case there is no useful information in the documents for the query, indicate in your answer that you are not able to answer the query. Remember that the user you're generating an answer for has no access to the documents. Do not reference the documents in your answer. There might be multiple relevant documents containing slightly different information, use all the relevant information available to produce the most accurate answer.
        """    
    data = dataset.get_dataset()
    
    queries = dataset.get_queries()
    preprocess = vectorizer.preprocess

    vector = vectorizer.make_vectorizer(False, preprocess) 

    X = vector.fit_transform(documents)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    model_knn.fit(X)

    for query_dict in queries['queries']:
        query = query_dict['q']
        input_string = prompt + "query: " + query + "\n"
        answer = query_dict['a']
        print("\n###########################################################################################################\n###########################################################################################################\n")
        
        query_proc = preprocess(query)
        query_vector = vector.transform([" ".join(query_proc)])
        distances, indices = model_knn.kneighbors(query_vector, n_neighbors=6)
        flattened_indices = indices.flatten()
        print(f'Query: {query}\n key words:: {query_proc}')
        print(f'documents: {list(flattened_indices)}')
        for i in (range(len(indices.flatten()))):
            input_string = input_string + "document " + str(i) + ": " + str(data[int(indices.flatten()[i])]) + "\n\n"

        input_string = input_string + "[/INST]"
        
        encoded_prompt = tokenizer(input_string, return_tensors="pt", add_special_tokens=False)
        encoded_prompt = encoded_prompt.to("cuda")
        
        generated_ids = model.generate(**encoded_prompt, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        print("\n" + decoded[0])

        print(f'\n expected: \n {answer}') 

# run the RAG system for a specifick query
def run_rag(prompt, query, model, tokenizer, documents):
    if not prompt:
        prompt=f"""
        [INST]<s>Your job is to answer the given query by leveraging the knowledge in the provided documents. Most of these documents will be relevant, but may also be irrelevant. Each document will have a distance listed with it, this indicates how relevant the document is. Try to use the documents with the lowest distance first. Only use information from relevant documents. Combine the information found in the different relevant documents and produce a compact, short and comprehensive answer to the query. In case there is no useful information in the documents for the query, indicate in your answer that you are not able to answer the query. Remember that the user you're generating an answer for has no access to the documents. Do not reference the documents in your answer. There might be multiple relevant documents containing slightly different information, use all the relevant information available to produce the most accurate answer.
        """
    keys = ["tags", "name", "steps"]
    data = dataset.get_dataset()

    documents, document_ids = dataset.make_documents(keys)
    queries = dataset.get_queries()
    preprocess = vectorizer.preprocess

    vector = vectorizer.make_vectorizer(False, preprocess) 

    X = vector.fit_transform(documents)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    model_knn.fit(X)

    input_string = prompt + "query: " + query + "\n"
    print("\n###########################################################################################################\n###########################################################################################################\n")
    
    query_proc = preprocess(query)
    query_vector = vector.transform([" ".join(query_proc)])
    distances, indices = model_knn.kneighbors(query_vector, n_neighbors=6)
    flattened_indices = indices.flatten()
    print(f'Query: {query}\n key words:: {query_proc}')
    print(f'documents: {list(flattened_indices)}')
    for i in (range(len(indices.flatten()))):
        input_string = input_string + "document " + str(i) + ": " + str(data[int(indices.flatten()[i])]) + "\n\n"

    input_string = input_string + "[/INST]"
    
    encoded_prompt = tokenizer(input_string, return_tensors="pt", add_special_tokens=False)
    encoded_prompt = encoded_prompt.to("cuda")
    
    generated_ids = model.generate(**encoded_prompt, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print("\n" + decoded[0])

#Query answering for the paper dataset
def run_rag_paper(prompt, query, queries, model, tokenizer, documents):
    if not prompt:
        prompt=f"""
        [INST]<s>Your job is to answer the given query by leveraging the knowledge in the provided documents. These documents might be relevant, but may also be irrelevant. Each document will have a distance listed with it, this indicates how relevant the document is. Try to use the documents with the lowest distance first. Only use information from relevant documents. Combine the information found in the different relevant documents and produce a compact, short and comprehensive answer to the query. In case there is no useful information in the documents for the query, indicate in your answer that you are not able to answer the query. Remember that the user you're generating an answer for has no access to the documents. Do not reference the documents in your answer. There might be multiple relevant documents containing slightly different information, use all the relevant information available to produce the most accurate answer.
        """
    if isinstance(queries, dict) and 'queries' in queries:
        queries = queries['queries']

    for query_dict in queries:
        if isinstance(query_dict, dict) and 'q' in query_dict:
            query = query_dict['q']
            answer = query_dict.get('a', "no answer")
        else:
            query = query_dict
            answer = "no answer"
        input_string = prompt + "query: " + query + "\n"
       
    keys = ['title', 'abstract', 'author']
    queries, anthology_sample = dataset.get_dataset_paper()

    #documents = dataset.make_documents_paper(keys, anthology_sample)
    preprocesed_docs = [neural.preprocess(doc) for doc in documents]

    embed_model = neural.get_neural_embedder()
    document_embeddings = embed_model.encode(preprocesed_docs)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    model_knn.fit(document_embeddings)

    input_string = prompt + "query: " + query + "\n"
    print("\n###########################################################################################################\n###########################################################################################################\n")
    
    query_proc = neural.preprocess(query)
    query_vector = embed_model.encode(query_proc)
    query_vector = query_vector.reshape(1, -1)
    distances, indices = model_knn.kneighbors(query_vector, n_neighbors=6)
    flattened_indices = indices.flatten()
    print(f'Query: {query}\n key words:: {query_proc}')
    print(f'documents: {list(flattened_indices)}')
    for i in (range(len(indices.flatten()))):
            input_string = input_string + "document " + str(i) + ", Distance : " + str(distances.flatten()[i]) + ": " + str(documents[int(indices.flatten()[i])]) + "\n\n"

    input_string = input_string + "[/INST]"
    
    encoded_prompt = tokenizer(input_string, return_tensors="pt", add_special_tokens=False)
    encoded_prompt = encoded_prompt.to("cuda")
    
    generated_ids = model.generate(**encoded_prompt, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print("\n" + decoded[0])


