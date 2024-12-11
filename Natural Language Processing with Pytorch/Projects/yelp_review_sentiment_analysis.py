import collections
import pandas as pd
import numpy as np
import re
from argparse import Namespace
import string
from datasets import Dataset, DatasetDict, load_dataset


#Importing the dataset

dataset = load_dataset("yelp_review_full")

# Convert to pandas DataFrame
train_csv = pd.DataFrame(dataset['train'])
test_csv = pd.DataFrame(dataset['test'])



train_csv = train_csv.rename(columns={"label": "rating",'text':'review'})
test_csv = test_csv.rename(columns={"label": "rating",'text':'review'})

print(train_csv)
print(test_csv)


#Partitioning the dataset

print(train_csv)
print(test_csv)


def partition_dataset(train_csv, test_csv):
  # Creating new train, validation, and test sets
  by_ratings = collections.defaultdict(list)
  for _, row in train_csv.iterrows():
      # Accessing 'rating' using the column name
      by_ratings[row['rating']].append(row.to_dict())
  final_list = []
  seed = 1000
  np.random.seed(seed)
  for _,item_list in sorted(by_ratings.items()):
    np.random.shuffle(item_list)
    total_rows = len(item_list)
    total_train_required = int(0.8*total_rows)
    total_test_required = int(0.2*total_rows)

    #given data pioint to split attribute
    for item in item_list[:total_train_required]:
      item['split'] = 'train'
    for item in item_list[total_train_required:total_train_required+total_test_required]:
      item['split'] = 'val'
    #add to final list
    final_list.extend(item_list)
  for _,row in test_csv.iterrows():
    row_dict = row.to_dict()
    row_dict['split'] = 'test'
    final_list.append(row_dict)

  return final_list, pd.DataFrame(final_list)



def preprocess_data(text):
  if type(text) == float:
        print(text)
  text = text.lower()
  text = re.sub(r"([.,!?])", r" \1 ", text)
  text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
  return text

final_list, final_list_df = partition_dataset(train_csv,test_csv)
final_list_df.review = final_list_df.review.apply(preprocess_data)


# see the split of data in the dataset
print("Final List DF",final_list)
final_list_df.split.value_counts()
print(final_list)


#Converting ratings to sentinement
final_list_sentiment = final_list_df.copy()
final_list_rating = final_list_df.copy()
print(final_list_sentiment.rating.unique())
final_list_sentiment['rating'] = final_list_sentiment.rating.apply({1:'negative',2:'positive'}.get)
print(final_list_sentiment.rating.unique())


import arg
# save the new dataframes to the csv file
final_list_rating.to_csv(arg.output_file_rating)
final_list_sentiment.to_csv(arg.output_file_sentiment)


class Vocabulary(object):
  """
Class to process text and extract vocabulary for mapping
  """
def __init__(self,token_to_idx=None,add_unk=True,unk_token="<UNK>"):
      if token_to_idx is None:
        token_to_idx = {}
      self._token_to_idx = token_to_idx
      self._idx_to_token = {idx:token for token,idx in self._token_to_idx.items()}
      self._add_unk  = add_unk
      self._unk_token = unk_token
      self.unk_index = -1
      if add_unk:
        self.unk_index = self.add_token(unk_token)

def to_serializable(self):
  "Returns a dictionary which can be serializable"
  return {'token_to_idx':self._token_to_idx,'add_unk':self._add_unk,
          'unk_token':self._unk_token}

@classmethod
def from_serializable(cls,contents):
  "Instantitate a vocab from serialized dictionary"
  return cls(**contents)

def add_token(self,token):
  """Update the mapping dictionary based on the token"""
  if token in self._token_to_idx:
    index = self._token_to_idx[token]
  else:
    index = len(self._token_to_idx)
    self._token_to_idx[token] = index
    self._token_to_idx[index] = token
  return index


def lookup_token(self,token):
  "Retreive the token based on the index from local mapping"
  if self.unk_index >= 0:
    return self._token_to_idx.get(token,self.unk_index)
  else:
    return self._token_to_idx[token]


def lookup_index(self, index):
  "retrieve the token based on the index from the mapping dictionary"
  if index not in self._idx_to_token:
    raise KeyError("The provided index: %d is not in the vocab"% index)
  else:
    return self._idx_to_token[index]

def __str__(self):
  return "<Vocabulary(size=%d)>" % len(self)


def __len__(self):
  "length of the vocabulary"
  return len(self._token_to_idx)



class ReviewVectorizer(object):
  " Used the vocbulary class to convert the tokens into actual numerical vector"

  def __init__(self, review_vocab, rating_vocab):
    """
    review_vocab :  maps word to integer
    rating_vocab : maps class label to integer("negative/positive")
    """
    self.review_vocab = review_vocab
    self.rating_vocab = rating_vocab

  def vectorize(self, review):
    "vectorize a text review to one hot encoding"
    one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)
    for token in review.split(" "):
      if token not in string.punctuation:
        one_hot[self.review_vocab.lookup_token(token)] = 1
    return one_hot



  @classmethod
  def from_dataframe(cls, review_df, cutoff=25):
    "instantiate a vector for reviews directly from dataset dataframe"
    review_vocab = Vocabulary(add_unk=True)
    rating_vocab = Vocabulary(add_unk=False)
    for rating in sorted(set(review_df.rating)):
      rating_vocab.add_token(rating)
    
    # add top words if count > provided counts
    word_count = collections.Counter()
    for review in review_df.review:
      for word in review.split(" "):
        if word not in string.punctuation:
          word_count[word] += 1


    for word, count in word_count.items():
      if count > cutoff:
        review_vocab.add_token(word)
    return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents):
      review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
      rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])
      return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)


    def to_serializable(self):
      return {'review_vocab' : self.review_vocab.to_serializable(),
              'rating_vocab' : self.rating_vocab.to_serializable()         
      }