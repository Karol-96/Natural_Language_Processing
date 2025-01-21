import pandas as pd
import torch 
import torch.nn as nn #neural network modules for LSTM & Embedding
import torch.optim as optim #For optimizing algorithims like Adam
from torch.utils.data import DataLoader, Dataset #Dataset and DataLoader for batching
from collections import Counter #For building vocabulary for Text Data
import re #Regular Expression for text processing

#Let's load and process data
data_path = './Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
data = pd.read_csv(data_path)

#Let's check dataset columns to understand it's structure
print(data.columns)

# Extract input (instruction) and output (response) pairs for chatbot training
input_texts = data['instruction'].fillna('')  # Replace missing instructions with an empty string
output_texts = data['response'].fillna('')  # Replace missing responses with an empty string


def preprocess_text(text):
    text = text.lower()  # Convert all text to lowercase for uniformity
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters, keeping alphanumeric and spaces
    return text

# Apply preprocessing to both input and output texts
input_texts = input_texts.apply(preprocess_text)
output_texts = output_texts.apply(preprocess_text)


#Function to build vocanulary from text corpus
def build_vocab(texts):
    counter = Counter() #For counting word frequencies
    for text in texts: 
        counter.update(text.split('')) #Split texts into word and count
    vocab = {word:idx + 1 for idx,word in enumerate(counter) } #Assigning unique index starting from 1
    vocab['<PAD>'] = 0 #Padding token for sequence alignment
    vocab['<SOS>'] = len(vocab) + 1 #Special start of sequence token
    vocab['<EOS>'] = len(vocab) + 2 #Special end of sequence token
    return vocab

#Build Vocabularies for Input and Outout texts
input_texts = build_vocab(input_texts)
output_texts = build_vocab(output_texts)

#Function to tokenize text using the vocabulary
def tokenize(text,vocab):
    tokens = [vocab['<SOS>']] + [vocab.get(word,0) for word in text.split()] + [vocab['<EOS>']]
    return tokens


input_sequences = [tokenize(text, input_vocab) for text in input_texts]
output_sequences = [tokenize(text, output_vocab) for text in output_texts]




