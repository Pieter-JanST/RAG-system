from transformers import pipeline, PegasusTokenizer, PegasusForConditionalGeneration, BartForConditionalGeneration, BartTokenizer
from datasets import Dataset
import textwrap
from torch.utils.data import DataLoader
import time
from summarizer import Summarizer
import nltk
import pickle
import dataset
import torch

"""
File containing different functions for compressing documents
"""

def get_summirizer():
    sum_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
    tokenizer =PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    return sum_model, tokenizer

def make_dataset():
    queries, anthology_sample = dataset.get_dataset_paper()

    documents = dataset.make_documents_paper(['title', 'full', 'author'], anthology_sample)
    data = {"text": documents}

    dataset = Dataset.from_dict(data)
    return dataset

def summarize_document_pegasus(doc, sum_model, tokenizer):
    if len(doc) < 4098:
        return doc
    # code inspiration adapted from https://discuss.huggingface.co/t/summarization-on-long-documents/920/6
    nested = []
    sent = []
    length = 0
    for sentence in nltk.sent_tokenize(doc):
        length += len(sentence)
        if length < 1024:
          sent.append(sentence)
        else:
          nested.append(sent)
          sent = []
          length = 0
    
    if sent:
        nested.append(sent)

    # Summarize each chunk and concatenate the summaries
    summaries = []
    for chunk in nested:
        tokenized = tokenizer.encode(''.join(chunk), truncation=True, return_tensors='pt')
        tokenized = tokenized.to("cuda")
        max_length= 40
        min_length= 10
        summary = sum_model.to('cuda').generate(tokenized, max_length=max_length, min_length=min_length, length_penalty=3.0)
        output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary]
        summaries.append(output)

    final_output = []
    for sublist in summaries:
        for item in sublist:
            final_output.append(item)
    
    # Join the sentences into a single string
    final_string = ' '.join(final_output)
    return final_string

def group_sentences(doc, n):
    sentences = nltk.sent_tokenize(doc)
    return [sentences[i:i+n] for i in range(0, len(sentences), n)]

def batch_processing_bert(batch, summarizer):
    summaries = []
    for doc in batch['text']:
        if len(doc) < 4000:
            summaries.append(doc)
        else:
            summaries.append(summarizer(doc, num_sentences=10))
    
    batch['text'] = summaries
    return batch

def batch_processing_pegasus(batch, sum_model, tokenizer):
    summaries = [summarize_document_pegasus(doc) for doc in batch['text']]
    batch['text'] = summaries
    return batch

# This takes about an hour on my hardware
def get_summaries_bert():
    summarizer = Summarizer()

    dataloader = DataLoader(make_dataset(), batch_size=32)

    summaries = []

    start = time.time()
    for i, batch in enumerate(dataloader):
        batch = batch_processing_bert(batch, summarizer)
        print(f'Done simmarizing batch: {i}/36, took: {(time.time() - start)/60} minutes')
        summaries.extend(batch['text'])

    summary_dataset = Dataset.from_dict({'text': summaries})

    with open('summary_dataset', 'wb') as output:
        pickle.dump(summary_dataset, output)

    return summary_dataset

# This takes about 3 hours on my hardware
def get_summaries_pegasus():
    dataloader = DataLoader(make_dataset(), batch_size=32)
    sum_model, tokenizer = get_summirizer()
    summaries = []

    start = time.time()
    for i, batch in enumerate(dataloader):
        batch = batch_processing_pegasus(batch,sum_model, tokenizer)
        print(f'Done simmarizing batch: {i}/36, took: {(time.time() - start)/60} minutes')
        summaries.extend(batch['text'])

    summary_dataset = Dataset.from_dict({'text': summaries})

    with open('summary_dataset_pegasus.pickle', 'wb') as output:
       pickle.dump(summary_dataset, output)

    return summary_dataset
   