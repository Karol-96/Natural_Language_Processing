import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


"""
Topic decomposition is a way for computers to look at a large set of text (like news articles, reviews, or books) and figure out the "hidden" topics people are talking about. 
These topics are groups of related words that often appear together.

Why is Topic Decomposition Needed?
1) Finding Structure in Chaos: Text data is messy, and topic decomposition helps organize it into meaningful groups.
2) Summarizing Large Texts: It helps reduce hundreds or thousands of documents into just a handful of key themes.



Real-World Applications:
1) Customer Feedback: A company can analyze reviews to find out what topics customers talk about (e.g., "product quality," "delivery issues").
2) News Aggregation: Grouping articles into topics like "politics," "sports," or "technology."
3) Healthcare Research: Finding themes in medical studies, like "cancer treatments" or "mental health."

Techniques for Topic Modeling
SVD (Singular Value Decomposition), also called Latent Semantic Analysis (LSA):

Think of it like squashing a tall stack of Lego pieces into a neat pile where the colors (words) most often used together are grouped.
How it works:
Breaks down the large matrix (document-term matrix: rows = documents, columns = words) into smaller parts.
Finds patterns by identifying which words tend to appear together in the same documents.
Why it's useful: It uncovers hidden relationships between words and topics, even if the exact word isn't used.
Real-World Example: Imagine you have customer reviews with words like "fast delivery," "speedy service," and "quick shipment." SVD might group these into a "delivery speed" topic even though the exact phrases differ.

NMF (Non-Negative Matrix Factorization):

This method focuses only on positive Lego pieces — it doesn’t allow negative relationships.
How it works:
Breaks the document-term matrix into two smaller matrices with only positive values.
These smaller matrices represent topics (groups of words) and their strength in each document.
Why it's useful: It produces more interpretable topics because it only considers additive relationships (e.g., words co-occurring).
Real-World Example: In a set of movie reviews, NMF might create topics like "romantic scenes" or "special effects" by clustering words like "love," "kiss," or "explosion," "graphics."


"""

class TopicModeling:
    def __init__(self,n_topics=5,n_top_words=10):
        """
        Initialize the Topic Modeling Class
        Args:
            n_topic(int) : Number of topics to extract
            n_top_words(int) : Number of top words to display per topic
        """
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words = 'english',
            max_df = 0.95,
            min_df = 2
        )
    
    def load_data(self):
        """
        Load and Prepare the dataset from sklearn: 20 newsgroups dataset
        """
        #Load Dataset
        print("Loading 20NewsGroup Dataset from Sklearn")
        dataset = fetch_20newsgroups(
            shuffle=True,
            random_state=42,
            remove=('headers','footers','quotes')
        )
        self.documents = dataset.data
        self.categories = dataset.target_names

        #Create document-term matrix using TFIDF 
        print('Creating TFIDF matrix ..')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self.terms = self.vectorizer.get_feature_names_out()

        print(f'Dataset shape: {self.tfidf_matrix.shape}')
        return self.tfidf_matrix
    
    def apply_svd(self):
        """
        Extract top words for each topic from SVD results
        """
        print("Applying SVD...") 
        self.svd = TruncatedSVD(n_components = self.n_topics, random_state = 42)
        self.svd_topics = self.svd.fit_transform(self.tfidf_matrix)

        #get variance explained
        explained_variance = self.svd.explained_variance_ratio_.sum()
        print(f"Variance explained by SVD: {explained_variance:.2%}")
        return self.get_svd_topics()

    
    def apply_nmf(self):
        """
        Apply Non-negative Matrix Factorization to document-term matrix
        """
        print('Applying NMF...')
        self.nmf = NMF(n_components=self.n_topics, random_state=42)
        self.nmf_topics = self.nmf.fit_transform(self.tfidf_matrix)

        return self.get_nmf_topics()
    def get_svd_topics(self):
        """
        Extract top words for each topic from SVD results
        """
        topic_words = {}
        for topic_idx, comp in enumerate(self.svd.components_):
            # Get top words based on component values
            word_idx = comp.argsort()[:-self.n_top_words-1:-1]
            topic_words[f"Topic {topic_idx+1}"] = [self.terms[i] for i in word_idx]
        
        return topic_words
    
    def get_nmf_topics(self):
        """
        Extract top words for each topic from NMF results
        """
        topic_words = {}
        for topic_idx, comp in enumerate(self.nmf.components_):
            # Get top words based on component values
            word_idx = comp.argsort()[:-self.n_top_words-1:-1]
            topic_words[f"Topic {topic_idx+1}"] = [self.terms[i] for i in word_idx]
        
        return topic_words
    def visualize_topics(self, topic_words, title):
        """
        Create a heatmap visualization of topic-word realtionships
        """
        #Create word importance matrix
        word_importance = np.zeros((self.n_topics, self.n_top_words))
        all_words = []

        for topic_idx, (topic,words) in enumerate(topic_words.items()):
            all_words.extend(words)
            for word_idx, word in enumerate(words):
                word_importance[topic_idx,word_idx] = 1

        #create heatmap
        plt.figure(figsize=(15,8))
        sns.heatmap(word_importance,
                    xticklabels=list(set(all_words)),
                    yticklabels=list(topic_words.keys()),
                    cmap = 'YlOrRd'
                    
                    )
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def analyze_document(self,text):
        """
        Analyze a new document using both SVD and NMF models

        Args:
                text(str) : Document text to analyze
        """

        #Transform document using TF-IDF
        doc_tfidf = self.vectorizer.transform([text])

        #Get topic distributions
        svd_dist = self.svd.transform(doc_tfidf)[0]
        nmf_dist = self.nmf.transform(doc_tfidf)[0]

        print('"\n Document topic Analysis:')
        print("\n SVD Topic Analysis:")
        for idx,value in enumerate(svd_dist):
            print(f'Topic {idx+1}: {value:.3f}')
        
        print(f'\n NMF Distribution:')
        for idx,value in enumerate(nmf_dist):
            print(f'Topic {idx+1}: {value:.3f}')


def main():
    #initiazlie and run topic modeling
    topic_modeler = TopicModeling(n_topics=5, n_top_words=10)

    #Load and prepare data
    topic_modeler.load_data()

    #Apply SVD and get topics
    svd_topics = topic_modeler.apply_svd()
    print('\SVD Topics:')
    for topic,words in svd_topics.items():
        print(f"{topic}: {','.join(words)}")

    #Apply NMF and get topics 
    nmf_topics = topic_modeler.apply_nmf()
    print(f'\n NMF Topics:')
    for topic,words in nmf_topics.items():
        print(f"{topic}: {','.join(words)}")

    #Visualize topics 
    topic_modeler.visualize_topics(svd_topics, "SVD Topic-Word Relationships")
    topic_modeler.visualize_topics(nmf_topics, "NMF Topic-Word Relationships")

    # Example: Analyze a new document
    sample_text = """
    The new computer system uses advanced algorithms and machine learning
    to process large amounts of data. The software is designed to be
    user-friendly and efficient.
    """
    
    topic_modeler.analyze_document(sample_text)

if __name__ == "__main__":
    main()