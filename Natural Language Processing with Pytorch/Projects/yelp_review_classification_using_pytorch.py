# Import required libraries
import torch  # PyTorch deep learning library
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Neural network functions
import torch.optim as optim  # Optimization algorithms
from torch.utils.data import Dataset, DataLoader  # Dataset handling utilities
from tqdm import tqdm_notebook  # Progress bar
import os  # Operating system interface
from collections import Counter  # For counting occurrences
import collections  # Additional data structures
import pandas as pd  # Data manipulation library
import numpy as np  # Numerical computing library
import re  # Regular expressions
from argparse import Namespace  # Command line argument parsing
import string  # String operations
from datasets import load_dataset  # HuggingFace datasets library

# Load the Yelp review dataset from HuggingFace
dataset = load_dataset("yelp_review_full")

# Convert dataset splits to pandas DataFrames
train_csv = pd.DataFrame(dataset['train'])
test_csv = pd.DataFrame(dataset['test'])

# Rename columns to match our expected format
train_csv = train_csv.rename(columns={"label": "rating",'text':'review'})
test_csv = test_csv.rename(columns={"label": "rating",'text':'review'})

def partition_dataset(train_csv, test_csv):
    """
    Split the dataset into train, validation and test sets while maintaining class balance
    Args:
        train_csv: Training data DataFrame
        test_csv: Test data DataFrame
    Returns:
        Tuple of (list of all data, DataFrame of all data)
    """
    # Group reviews by rating
    by_ratings = collections.defaultdict(list)
    for _, row in train_csv.iterrows():
        by_ratings[row['rating']].append(row.to_dict())
    final_list = []
    seed = 1000
    np.random.seed(seed)
    
    # Split each rating group into train/val with 80/20 ratio
    for _, item_list in sorted(by_ratings.items()):
        np.random.shuffle(item_list)
        total_rows = len(item_list)
        total_train_required = int(0.8*total_rows)
        total_test_required = int(0.2*total_rows)

        for item in item_list[:total_train_required]:
            item['split'] = 'train'
        for item in item_list[total_train_required:total_train_required+total_test_required]:
            item['split'] = 'val'
        final_list.extend(item_list)
        
    # Add test data
    for _, row in test_csv.iterrows():
        row_dict = row.to_dict()
        row_dict['split'] = 'test'
        final_list.append(row_dict)

    return final_list, pd.DataFrame(final_list)

def preprocess_text(text):
    """
    Clean and normalize text data
    Args:
        text: Input text string
    Returns:
        Preprocessed text string
    """
    if isinstance(text, float):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"([.,!?])", r" \1 ", text)  # Add spaces around punctuation
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)  # Remove non-alphanumeric chars
    return text

class Vocabulary(object):
    """
    Vocabulary class to convert tokens to indices and vice versa
    Handles unknown tokens with special <UNK> token
    """
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """Convert vocabulary to JSON serializable format"""
        return {
            "token_to_idx": self._token_to_idx,
            "add_unk": self._add_unk,
            "unk_token": self._unk_token
        }

    @classmethod
    def from_serializable(cls, contents):
        """Create vocabulary from serialized format"""
        return cls(**contents)

    def add_token(self, token):
        """Add a token to vocabulary"""
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        """Convert token to index"""
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """Convert index to token"""
        if index not in self._idx_to_token:
            raise KeyError(f"the index ({index}) is not in the Vocabulary")
        return self._idx_to_token[index]

    def __len__(self):
        return len(self._token_to_idx)

class ReviewVectorizer(object):
    """
    Converts text reviews into numerical vectors for model input
    Uses one-hot encoding for the review text
    """
    def __init__(self, review_vocab, rating_vocab):
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        """Convert a review text into one-hot encoded vector"""
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)
        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1
        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """Create vectorizer from DataFrame with frequency cutoff for vocabulary"""
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)
        
        # Add all unique ratings to vocabulary
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        # Count word frequencies
        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        # Add words that appear more than cutoff times
        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)
                
        return cls(review_vocab, rating_vocab)

class ReviewDataset(Dataset):
    """
    PyTorch Dataset class for Yelp reviews
    Handles train/val/test splits and vectorization
    """
    def __init__(self, review_df, vectorizer):
        self.review_df = review_df
        self.vectorizer = vectorizer
        
        # Split data into train/val/test
        self.train_df = self.review_df[self.review_df.split=='train']
        self.train_size = len(self.train_df)
        self.val_df = self.review_df[self.review_df.split=='val']
        self.validation_size = len(self.val_df)
        self.test_df = self.review_df[self.review_df.split=='test']
        self.test_size = len(self.test_df)
        
        # Create lookup dictionary for splits
        self._lookup_dict = {'train': (self.train_df, self.train_size),
                            'val': (self.val_df, self.validation_size),
                            'test': (self.test_df, self.test_size)}
        self.set_split('train')

    def set_split(self, split="train"):
        """Set the current data split to use"""
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """Get a single sample from dataset"""
        row = self._target_df.iloc[index]
        review_vector = self.vectorizer.vectorize(row.review)
        rating_index = self.vectorizer.rating_vocab.lookup_token(row.rating)
        return {'x_data': review_vector,
                'y_target': rating_index}

    def get_num_batches(self, batch_size):
        """Calculate number of batches per epoch"""
        return len(self) // batch_size

class ReviewClassifier(nn.Module):
    """
    Simple neural network classifier for reviews
    Single fully connected layer with sigmoid activation
    """
    def __init__(self, num_features):
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)
    
    def forward(self, x_in, apply_sigmoid=False):
        """Forward pass through the network"""
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out

# Setup hyperparameters and configuration
args = Namespace(
    frequency_cutoff=25,  # Minimum word frequency to include in vocabulary
    model_state_file="model.pth",  # Model checkpoint file
    save_dir="model_storage",  # Directory to save model
    vectorizer_file="vectorizer.json",  # Vectorizer save file
    batch_size=128,  # Number of samples per batch
    early_stopping_criteria=5,  # Epochs to wait before early stopping
    learning_rate=0.001,  # Learning rate for optimizer
    num_epochs=100,  # Maximum number of training epochs
    seed=42,  # Random seed for reproducibility
    cuda=torch.cuda.is_available(),  # Use GPU if available
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# CUDA setup - use GPU if available
if not torch.cuda.is_available():
    args.cuda = False
print(f"Using CUDA: {args.cuda}")
args.device = torch.device("cuda" if args.cuda else "cpu")

# Set random seeds for reproducibility
def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

set_seed_everywhere(args.seed, args.cuda)

# Create directories if needed
def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

handle_dirs(args.save_dir)

# Initialize training state dictionary
def make_train_state(args):
    """Initialize dictionary to track training state"""
    return {
        "stop_early": False,
        "early_stopping_step": 0,
        "early_stopping_best_val": 1e8,
        "learning_rate": args.learning_rate,
        "epoch_index": 0,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": -1,
        "test_acc": -1,
        "model_filename": args.model_state_file
    }

def update_train_state(args, model, train_state):
    """Update training state and handle early stopping"""
    # Save first model
    if train_state["epoch_index"] == 0:
        torch.save(model.state_dict(), train_state["model_filename"])
        train_state["stop_early"] = False
    # Save model if performance improved
    elif train_state["epoch_index"] >= 1:
        loss_tm1, loss_t = train_state["val_loss"][-2:]
        
        # Update early stopping counter
        if loss_t >= train_state["early_stopping_best_val"]:
            train_state["early_stopping_step"] += 1
        else:
            if loss_t < train_state["early_stopping_best_val"]:
                torch.save(model.state_dict(), train_state["model_filename"])
            train_state["early_stopping_step"] = 0
            
        # Stop early if patience is exceeded
        train_state["stop_early"] = \
            train_state["early_stopping_step"] >= args.early_stopping_criteria
            
    return train_state

def compute_accuracy(y_pred, y_target):
    """Calculate classification accuracy"""
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).long()
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

# Helper function to generate batches
def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    """Create batches of data from dataset"""
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=shuffle, drop_last=drop_last)
    
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

# Initialize dataset and model
if args.reload_from_files:
    print("Loading the Dataset and Vectorizer")
    dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv, 
                                                           args.vectorizer_file)
else:
    print("Loading the Dataset and Creating the Vectorizer")
    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
    dataset.save_vectorizer(args.vectorizer_file)

vectorizer = dataset.get_vectorizer()
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))

# Setup training components
classifier = classifier.to(args.device)
loss_func = nn.BCEWithLogitsLoss()  # Binary Cross Entropy loss
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(  # Learning rate scheduler
    optimizer=optimizer,
    mode="min",
    factor=0.5,
    patience=1
)

train_state = make_train_state(args)

# Setup progress bars
epoch_bar = tqdm.notebook.tqdm(
    desc="training routine",
    total=args.num_epochs,
    position=0
)

dataset.set_split("train")
train_bar = tqdm.notebook.tqdm(
    desc="split=train",
    total=dataset.get_num_batches(args.batch_size),
    position=1,
    leave=True
)

dataset.set_split("val")
val_bar = tqdm.notebook.tqdm(
    desc="split=val",
    total=dataset.get_num_batches(args.batch_size),
    position=1,
    leave=True
)

# Main training loop
try:
    for epoch_index in range(args.num_epochs):
        train_state["epoch_index"] = epoch_index
        
        # Training phase
        dataset.set_split("train")
        batch_generator = generate_batches(dataset, batch_size=args.batch_size, 
                                        device=args.device)
        
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
        
        for batch_index, batch_dict in enumerate(batch_generator):
            # Step 1: Zero gradients
            optimizer.zero_grad()
            
            # Step 2: Compute the output
            y_pred = classifier(x_in=batch_dict["x_data"].float())
            
            # Step 3: Compute the loss
            loss = loss_func(y_pred, batch_dict["y_target"].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            
            # Step 4: Use loss to produce gradients
            loss.backward()
            
            # Step 5: Use optimizer to take gradient step
            optimizer.step()
            
            # Compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            
            # Update bar
            train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                epoch=epoch_index)
            train_bar.update()
            
        train_state["train_loss"].append(running_loss)
        train_state["train_acc"].append(running_acc)
        
        # Iterate over val dataset
        dataset.set_split("val")
        batch_generator = generate_batches(dataset, batch_size=args.batch_size, 
                                        device=args.device)
        running_loss = 0.
        running_acc = 0.
        classifier.eval()
        
        for batch_index, batch_dict in enumerate(batch_generator):
            # Compute the output
            y_pred = classifier(x_in=batch_dict["x_data"].float())
            
            # Compute the loss
            loss = loss_func(y_pred, batch_dict["y_target"].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            
            # Compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            
            val_bar.set_postfix(loss=running_loss, acc=running_acc, 
                              epoch=epoch_index)
            val_bar.update()
            
        train_state["val_loss"].append(running_loss)
        train_state["val_acc"].append(running_acc)
        
        train_state = update_train_state(args=args, model=classifier, 
                                       train_state=train_state)
        scheduler.step(train_state["val_loss"][-1])
        
        if train_state["stop_early"]:
            break
            
        train_bar.n = 0
        val_bar.n = 0
        epoch_bar.update()
        
except KeyboardInterrupt:
    print("Exiting loop")

# Evaluation and prediction functions
def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    """Predict the rating for a review"""
    review = preprocess_text(review)
    vectorized_review = torch.tensor(vectorizer.vectorize(review))
    result = classifier(vectorized_review.view(1, -1))
    probability_value = F.sigmoid(result).item()
    index = 1 if probability_value >= decision_threshold else 0
    return vectorizer.rating_vocab.lookup_index(index)

# Test the model
classifier.load_state_dict(torch.load(train_state["model_filename"]))
classifier = classifier.to(args.device)
dataset.set_split("test")
batch_generator = generate_batches(dataset, batch_size=args.batch_size, 
                                 device=args.device)

running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # Compute the output
    y_pred = classifier(x_in=batch_dict["x_data"].float())
    # Compute the loss
    loss = loss_func(y_pred, batch_dict["y_target"].float())
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)
    # Compute the accuracy
    acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
    running_acc += (acc_t - running_acc) / (batch_index + 1)

train_state["test_loss"] = running_loss
train_state["test_acc"] = running_acc

print(f"Test loss: {train_state['test_loss']:.3f}")
print(f"Test accuracy: {train_state['test_acc']:.3f}")

# Example prediction
review = "I am annoyed with the Movie."
prediction = predict_rating(review, classifier, vectorizer, decision_threshold=0.5)
print(f"{review}: {prediction}")