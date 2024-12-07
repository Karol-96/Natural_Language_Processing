import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from random import shuffle
import string


#Download required NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')


class DocumentClassifier:
    def __init__(self):
        """Initialize the Document Classifier with necessary components"""
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.punctuation = string.punctuation
        self.classifier = None
    def preprocess_text(self, text):
        """
        Preprocess the text by:
        1. Converting to lowercase
        2. Tokenizing
        3. Removing stopwords and punctuation
        4. Creating a frequency distribution of words
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            dict: Word frequency dictionary
        """
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and punctuation
        tokens = [token for token in tokens 
                 if token not in self.stopwords 
                 and token not in self.punctuation]
        
        # Create word frequency dictionary
        word_freq = FreqDist(tokens)
        return {word: True for word in word_freq}
    def prepare_movie_review_data(self):
        """
        Prepare movie review dataset for classification
        
        Returns:
            tuple: (training_data, testing_data)
        """
        # Get positive and negative reviews
        positive_fileids = movie_reviews.fileids('pos')
        negative_fileids = movie_reviews.fileids('neg')
        
        # Prepare features for positive reviews
        positive_features = [(self.preprocess_text(movie_reviews.raw(fileids=fileid)), 'positive')
                           for fileid in positive_fileids]
        
        # Prepare features for negative reviews
        negative_features = [(self.preprocess_text(movie_reviews.raw(fileids=fileid)), 'negative')
                           for fileid in negative_fileids]
        
        # Combine and shuffle the features
        all_features = positive_features + negative_features
        shuffle(all_features)
        
        # Split into training (80%) and testing (20%) sets
        split_point = int(len(all_features) * 0.8)
        train_set = all_features[:split_point]
        test_set = all_features[split_point:]
        
        return train_set, test_set
    def train_naive_bayes(self, train_set):
        """
        Train a Naive Bayes classifier
        
        Args:
            train_set (list): Training data
            
        Returns:
            NaiveBayesClassifier: Trained classifier
        """
        self.classifier = NaiveBayesClassifier.train(train_set)
        return self.classifier
    def train_decision_tree(self, train_set):
        """
        Train a Decision Tree classifier
        
        Args:
            train_set (list): Training data
            
        Returns:
            DecisionTreeClassifier: Trained classifier
        """
        self.classifier = DecisionTreeClassifier.train(train_set)
        return self.classifier
    def evaluate_classifier(self, test_set):
        """
        Evaluate the classifier's performance
        
        Args:
            test_set (list): Testing data
            
        Returns:
            float: Accuracy score
        """
        return accuracy(self.classifier, test_set)
    
    def classify_text(self, text):
        """
        Classify a new text document
        
        Args:
            text (str): Text to classify
            
        Returns:
            str: Predicted class label
        """
        features = self.preprocess_text(text)
        return self.classifier.classify(features)
def main():
    """Main function to demonstrate document classification"""
    
    print("=== Movie Review Sentiment Classification ===")
    
    # Initialize classifier
    doc_classifier = DocumentClassifier()
    
    # Prepare data
    print("\nPreparing movie review data...")
    train_set, test_set = doc_classifier.prepare_movie_review_data()
    print("Train Set:",train_set)
    print("Test Set:",test_set)
    
    # Train and evaluate Naive Bayes Classifier
    print("\nTraining Naive Bayes Classifier...")
    nb_classifier = doc_classifier.train_naive_bayes(train_set)
    nb_accuracy = doc_classifier.evaluate_classifier(test_set)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.2%}")
    
    # Show most informative features
    print("\nMost Informative Features:")
    nb_classifier.show_most_informative_features(5)
    
    # Train and evaluate Decision Tree Classifier
    print("\nTraining Decision Tree Classifier...")
    dt_classifier = doc_classifier.train_decision_tree(train_set)
    dt_accuracy = doc_classifier.evaluate_classifier(test_set)
    print(f"Decision Tree Accuracy: {dt_accuracy:.2%}")
    
    # Example classification
    sample_positive = """
    This movie was fantastic! The acting was great, and the plot kept me engaged throughout.
    The cinematography was beautiful, and the ending was perfect.
    """
    
    sample_negative = """
    Terrible waste of time. Poor acting, boring story, and awful direction.
    I couldn't wait for it to end. Don't waste your money on this one.
    """
    
    print("\nClassifying sample reviews:")
    print("Positive review classification:", doc_classifier.classify_text(sample_positive))
    print("Negative review classification:", doc_classifier.classify_text(sample_negative))

if __name__ == "__main__":
    main()