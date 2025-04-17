### QUESTION 1
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Make sure stopwords are available
nltk.download('stopwords', force=True)

# Load spaCy English model
import subprocess
import sys

# Install spaCy model if not already available
try:
    nlp = spacy.load("en_core_web_sm")
except:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def nlp_pipeline_spacy(sentence):
    doc = nlp(sentence)

    # 1. Original Tokens
    tokens = [token.text for token in doc]
    print("1. Original Tokens:")
    print(tokens)

    # 2. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens_without_stopwords = [token.text for token in doc if token.text.lower() not in stop_words and token.is_alpha]
    print("\n2. Tokens Without Stopwords:")
    print(tokens_without_stopwords)

    # 3. Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word.lower()) for word in tokens_without_stopwords]
    print("\n3. Stemmed Words:")
    print(stemmed_words)

# Example
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
nlp_pipeline_spacy(sentence)

#OUTPUT :
    
    [nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
1. Original Tokens:
['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']

2. Tokens Without Stopwords:
['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri']

3. Stemmed Words:
['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri']



###QUESION 2 

import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Input sentence
sentence = "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

# Process the sentence
doc = nlp(sentence)

# Extract and print named entities
for ent in doc.ents:
    print(f"Text: {ent.text}, Label: {ent.label_}, Start: {ent.start_char}, End: {ent.end_char}")

#OUTPUT:
Text: Barack Obama, Label: PERSON, Start: 0, End: 12
Text: 44th, Label: ORDINAL, Start: 27, End: 31
Text: the United States, Label: GPE, Start: 45, End: 62
Text: the Nobel Peace Prize, Label: WORK_OF_ART, Start: 71, End: 92
Text: 2009, Label: DATE, Start: 96, End: 100

###QUESTION 3
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    # Step 1: Dot product of Q and K transpose
    scores = np.dot(Q, K.T)

    # Step 2: Scale scores
    d_k = K.shape[1]  # key dimension
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: Apply softmax to get attention weights
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # for numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attention_weights = softmax(scaled_scores)

    # Step 4: Multiply weights by V
    output = np.dot(attention_weights, V)

    print("1. Attention Weights (after softmax):")
    print(attention_weights)

    print("\n2. Final Output (Attention x V):")
    print(output)

# Test inputs
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

scaled_dot_product_attention(Q, K, V)
#OUTPUT :
1. Attention Weights (after softmax):
[[0.73105858 0.26894142]
 [0.26894142 0.73105858]]

2. Final Output (Attention x V):
[[2.07576569 3.07576569 4.07576569 5.07576569]
 [3.92423431 4.92423431 5.92423431 6.92423431]]

###QUESTION 4 :
from transformers import pipeline

# Load a pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Input sentence
sentence = "Despite the high price, the performance of the new MacBook is outstanding."

# Get the sentiment prediction
result = sentiment_pipeline(sentence)[0]

# Print the sentiment label and confidence score
print(f"Sentiment: {result['label']}")
print(f"Confidence Score: {result['score']}")


#OUTPUT :
No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
config.json: 100%
 629/629 [00:00<00:00, 47.5kB/s]
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
WARNING:huggingface_hub.file_download:Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
model.safetensors: 100%
 268M/268M [00:01<00:00, 235MB/s]
tokenizer_config.json: 100%
 48.0/48.0 [00:00<00:00, 4.11kB/s]
vocab.txt: 100%
 232k/232k [00:00<00:00, 14.6MB/s]
Device set to use cuda:0
Sentiment: POSITIVE
Confidence Score: 0.9998302459716797

