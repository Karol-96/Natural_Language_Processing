import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd

class SimpleNLPNeuralNetwork:
    def __init__(self, max_words=10000, max_len=200):
        """
        Initialize the Neural Network for NLP
        
        Args:
            max_words: Maximum number of words to keep in vocabulary
            max_len: Maximum length of each sequence
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        
    def prepare_data(self, texts, labels):
        """
        Prepare text data for neural network processing
        """
        # Convert texts to sequences
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences to ensure uniform length
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        
        return padded_sequences, np.array(labels)
    
    def build_model(self, embedding_dim=100):
        """
        Build a simple neural network model for text classification
        """
        model = Sequential([
            # Embedding layer converts word indices to dense vectors
            Embedding(self.max_words, embedding_dim, input_length=self.max_len),
            
            # Global average pooling reduces sequence dimension
            GlobalAveragePooling1D(),
            
            # Dense layer with ReLU activation
            Dense(64, activation='relu'),
            
            # Output layer with sigmoid activation for binary classification
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def main():
    # Example movie reviews dataset
    reviews = [
        "This movie was fantastic! Great acting and storyline",
        "Terrible waste of time. Poor acting and boring plot",
        "I loved every minute of this film, highly recommended",
        "Don't waste your money, this movie is awful",
        "Amazing special effects and wonderful direction",
        # Add more reviews as needed
    ]
    
    # Labels: 1 for positive, 0 for negative
    labels = [1, 0, 1, 0, 1]
    
    # Initialize our NLP Neural Network
    nn_classifier = SimpleNLPNeuralNetwork()
    
    # Prepare the data
    X, y = nn_classifier.prepare_data(reviews, labels)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build and train the model
    model = nn_classifier.build_model()
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.2f}")
    
    # Example predictions
    new_reviews = [
        "This movie exceeded all my expectations!",
        "I fell asleep during the first hour, very boring"
    ]
    
    # Prepare new reviews for prediction
    new_sequences = nn_classifier.tokenizer.texts_to_sequences(new_reviews)
    new_padded = pad_sequences(new_sequences, maxlen=nn_classifier.max_len)
    
    # Make predictions
    predictions = model.predict(new_padded)
    
    print("\nPredictions for new reviews:")
    for review, pred in zip(new_reviews, predictions):
        sentiment = "Positive" if pred > 0.5 else "Negative"
        confidence = pred if pred > 0.5 else 1 - pred
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} (confidence: {confidence[0]:.2f})")

if __name__ == "__main__":
    main()