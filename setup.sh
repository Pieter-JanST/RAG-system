#!/bin/bash

# Install Python packages
pip install -r requirements.txt

# Install additional packages
pip install wget
pip install git+https://github.com/huggingface/transformers
pip install datasets bitsandbytes accelerate xformers einops
pip -q install datasets
pip install pynndescent
pip install sentence-transformers
pip install bert-extractive-summarizer

# Get dataset, queries, ...
# wget <Recipe dataset url>
# wget <Paper dataset url>

# 
# Download necessary NLTK packages
python3 -m nltk.downloader punkt
python3 -m nltk.downloader stopwords
python3 -m nltk.downloader wordnet
python3 -m nltk.downloader averaged_perceptron_tagger