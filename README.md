# Sentiment Analyzer

A Python program that detects whether a sentence is 
positive or negative using a pre-trained Transformer model.

Built as Project 2 of my AI portfolio.

## What it does
Takes any English sentence and returns:
- Whether it is POSITIVE or NEGATIVE
- How confident the model is (percentage)

## How it works
Uses DistilBERT — a smaller faster version of BERT — 
built on the Transformer architecture (Vaswani et al. 2017).
Pre-trained on millions of sentences. Uses attention mechanism 
to understand relationships between words before classifying.

## Limitations
- No neutral output — only POSITIVE or NEGATIVE
- Mixed sentiment sentences get forced into one category
- Neutral sentences like "the weather is okay" get 
  misclassified as strongly positive

## How to run
pip install transformers
python sentiment.py

## Built with
- Python 3.14
- HuggingFace Transformers
- DistilBERT pre-trained model
