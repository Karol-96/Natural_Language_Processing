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
input_vocab = build_vocab(input_texts)
output_vocab = build_vocab(output_texts)

#Function to tokenize text using the vocabulary
def tokenize(text,vocab):
    tokens = [vocab['<SOS>']] + [vocab.get(word,0) for word in text.split()] + [vocab['<EOS>']]
    return tokens


input_sequences = [tokenize(text, input_vocab) for text in input_texts]
output_sequences = [tokenize(text, output_vocab) for text in output_texts]

#Custom Dataset class to handle out tokenized data
class ChatDataset(Dataset):
    def __init__(self, input_sequences, output_sequences):
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self,idx):
        return torch.tensor(self.input_sequences[idx], dtype = torch.long), \
                torch.tensor(self.output_sequences[idx], dtype=torch.long)
    
#Create DataLoader to handle batching and shuffling of data
batch_size = 32 #Number of samples processed together in one batch
dataset = ChatDataset(input_sequences, output_sequences)
dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True, collate_fn = lambda x: zip(*x))




# --------------------------
# Step 2: Define the Seq2Seq Model with Attention
# --------------------------

#Encoder : Encode the input sequence into a context vector
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super(EncoderRNN,self).__init_()
        self.embedding = nn.Embedding(input_size, embed_size) #Embdedding layers for word vectors
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True) #LSTM to process sequences

    def forward(self,x):
        embedded = self.embedding(x) #convert tokens to embeddings
        outputs, (hidden,cell) = self.rnn(embedded) #LSTM Return outputs and hidden states
        return outputs, (hidden,cell) #outputs for attention , hidden/cell for decoder initlalization
    

                                      


# Attention: Computes the attention weights over encoder outputs
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, hidden_size) #Combine hidden and encoder output
        self.v = nn.Linear(hidden_size,1, bias = True) #Scalar attention score for each token
    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)  # Number of tokens in the sequence
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)  # Repeat decoder hidden state
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))  # Compute energy
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)  # Normalize scores
        return attention_weights

# Decoder: Decodes the context vector and generates the target sequence
class DecoderRNN(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)  # Embedding layer for target tokens
        self.rnn = nn.LSTM(hidden_size + embed_size, hidden_size, batch_first=True)  # LSTM for decoding
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer for final predictions
        self.attention = Attention(hidden_size)  # Attention mechanism

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)  # Add time dimension
        embedded = self.embedding(x)  # Embed target token
        attention_weights = self.attention(hidden[-1], encoder_outputs)  # Compute attention weights
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # Weighted sum of encoder outputs
        rnn_input = torch.cat((embedded, context), dim=2)  # Combine context and embedding
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # Decode
        predictions = self.fc(outputs.squeeze(1))  # Generate predictions
        return predictions, hidden, cell, attention_weights

# Seq2Seq: Combines Encoder and Decoder into a single model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        encoder_outputs, (hidden, cell) = self.encoder(src)  # Encode the input
        outputs = []
        x = trg[:, 0]  # Start with <SOS> token
        for t in range(1, trg.size(1)):
            output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs)  # Decode step
            outputs.append(output)
            x = trg[:, t]  # Teacher forcing: use ground truth token
        return torch.stack(outputs, dim=1)
    
# --------------------------
# Step 3: Model Training
# --------------------------

# Define hyperparameters
input_size = len(input_vocab)  # Vocabulary size for input
output_size = len(output_vocab)  # Vocabulary size for output
embed_size = 256  # Embedding dimension for words
hidden_size = 512  # LSTM hidden layer size
epochs = 10  # Number of training iterations
learning_rate = 0.001  # Optimizer learning rate

# Initialize model
encoder = EncoderRNN(input_size, embed_size, hidden_size)
decoder = DecoderRNN(output_size, embed_size, hidden_size)
model = Seq2Seq(encoder, decoder).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer for training
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Loss function, ignoring <PAD>



# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)  # Pad input sequences
        trg = torch.nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)  # Pad target sequences
        src, trg = src.to(model.device), trg.to(model.device)  # Move to device

        optimizer.zero_grad()  # Clear previous gradients
        output = model(src, trg)  # Forward pass
        output = output.reshape(-1, output.shape[2])  # Reshape for loss calculation
        trg = trg[:, 1:].reshape(-1)  # Ignore <SOS> in target
        loss = criterion(output, trg)  # Compute loss
        loss.backward()  # Backpropagate gradients
        optimizer.step()  # Update model parameters

        epoch_loss += loss.item()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}')


# --------------------------
# Step 4: Save Model for Deployment
# --------------------------

model_path = "chatbot_model.pth"  # File to save the trained model
torch.save(model.state_dict(), model_path)  # Save model weights
print(f"Model saved to {model_path}")






