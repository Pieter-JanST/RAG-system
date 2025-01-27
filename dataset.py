import json
import datasets
import nltk

# get the queries
def get_queries():
    return json.load(open("./queries.json", "r"))

# get the entire dataset
def get_dataset():
    return datasets.load_dataset("parquet", data_files="./recipes.indexed.parquet")['train']

# filter the documents on a specifick key
def make_documents(keys):
    dataset = get_dataset()
    documents = []
    document_ids = []
    for recipe in dataset:
        recipe_string =""
        for key in keys:
            s = recipe[key]
            if s is not None:
                recipe_string += s + " "
        documents.append(recipe_string)
        document_ids.append(recipe['official_id'])
    return documents, document_ids

def make_documents_paper(keys, dataset):
    documents = []
    for doc in dataset:
        text = doc['title']
        if 'abstract' in keys and doc['abstract']:
            text = text + ' ' + doc['abstract']
        if 'full' in keys and doc['full_text']:
            text = text + ' ' + doc['full_text']
        if 'author' in keys and doc['author']:
            text = text + ' ' + doc['author']
        
        documents.append(text)
    return documents

def get_dataset_paper():
    SEED = 788663
    DOCS_TO_ADD = 1000

    query_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_queries.parquet")["train"]
    all_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_full.parquet")["train"]
    # Shuffle with seed and take only n docs
    random_documents = all_documents.shuffle(seed=SEED).take(DOCS_TO_ADD)
    # Concatenate relevant documents with random sample and shuffle again
    anthology_sample = datasets.concatenate_datasets([query_documents, random_documents]).shuffle(seed=SEED)
    # Export to Parquet to avoid downloading full anthology
    anthology_sample.to_parquet("./anthology_sample.parquet")

    queries = json.load(open("./acl_anthology_queries.json", "r"))
    return queries, anthology_sample
