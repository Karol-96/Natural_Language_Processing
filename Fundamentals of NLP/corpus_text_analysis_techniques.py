import nltk 
from nltk.corpus import webtext, brown, reuters, inaugural
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import ssl

# 1. Web and Chat Text Analysis
# Applications:
# Social media sentiment analysis
# Customer service chatbots
# Content moderation systems
# Online behavior analysis
# Marketing trend analysis
# Real Example: Companies like Facebook use chat analysis to detect hate speech, spam, or inappropriate content.


# Brown Corpus Analysis
# Applications:
# Genre classification
# Writing style analysis
# Plagiarism detection
# Author attribution
# Content recommendation systems
# Real Example: Amazon uses genre analysis to recommend books to readers based on writing style and content patterns.

# Reuters Corpus (News Analysis)
# Applications:
# News categorization
# Topic modeling
# Financial market analysis
# Trend prediction
# Automated news summarization
# Real Example: Bloomberg uses news analysis to provide real-time financial insights and market predictions.


# Inaugural Address Analysis

# Applications:
# Political discourse analysis
# Historical trend analysis
# Speech writing assistance
# Policy change tracking
# Public sentiment analysis
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('webtext',quiet=True)
nltk.download('brown', quiet=True)
nltk.download('reuters', quiet=True)
nltk.download('inaugural', quiet=True)
# nltk.download('punkt')

def explore_corpus(corpus,title):
    """Helper function to explore basic corpus statistics"""
    print(f'\n====={title}=====')
    print(f'Number of words: {len(corpus.words())}')
    print(f"Number of unique words: {len(set(corpus.words()))}")
    print(f"First 5 words: {corpus.words()[:5]}")
    if hasattr(corpus,'categories'):
        print(f"Categories: {corpus.categories()[:5]}")


#Web and Chat Text
print("\n === Web and Chat Text Analysis ====")
files = webtext.fileids()
print(f'Available web text files: {files}')

#Analyze the chat message
chat = webtext.words('singles.txt')
chat_freq = FreqDist(word.lower() for word in chat)
print(f"\n Most common words in chat:")
print(chat_freq.most_common(10))



#2. Brown Corpus Analysis
print("\n==== Brown Corpus Analysis =====")
explore_corpus(brown, "Brown Corpus")

#Genre based analysis
genres = ['news','romance','science_fiction']
genre_words = [ (genre,word)
               for genre in genres
               for word in brown.words(categories =genre)
               
               ]
genre_dist = ConditionalFreqDist((genre,word.lower())
                                for genre,word in genre_words
                                )
#Plot word frequencies across genres
plt.figure(figsize=(12,6))
words_to_compare = ['the','love','space']
genre_dist.tabulate(conditions=genres,samples=words_to_compare)
plt.title("Word Frequencies Across Genres")
plt.tight_layout()
plt.savefig('genre_frequncies.png')



# 3. Reuters Corpus
print("\n=== Reuters Corpus Analysis ===")
explore_corpus(reuters, "Reuters Corpus")

# Analyze topics
topics = reuters.categories()
print(f"\nSample topics: {topics[:10]}")

# Topic-based analysis
def show_topic_stats(topic):
    files = reuters.fileids(categories=[topic])
    words = reuters.words(fileids=files)
    return len(files), len(words)

print("\nTopic Statistics:")
for topic in topics[:5]:
    files, words = show_topic_stats(topic)
    print(f"{topic}: {files} articles, {words} words")

# 4. Inaugural Address Corpus
print("\n=== Inaugural Address Analysis ===")
explore_corpus(inaugural, "Inaugural Address Corpus")

# Timeline analysis
def analyze_inaugural_timeline():
    cfd = ConditionalFreqDist(
        (target_year, word.lower())
        for target_year in inaugural.fileids()
        for word in inaugural.words(target_year)
    )
    
    # Compare word usage over time
    words_of_interest = ['america', 'democracy', 'freedom', 'war']
    years = [f for f in inaugural.fileids()][-5:]  # Last 5 addresses
    
    print("\nWord usage in recent inaugural addresses:")
    cfd.tabulate(conditions=years, samples=words_of_interest)

analyze_inaugural_timeline()

# 5. Generate Random Text with Bigrams
def generate_random_text(corpus, num_words=50):
    """Generate random text using bigrams from the corpus"""
    import random  # Add this import
    
    words = corpus.words()
    bigrams = list(nltk.bigrams(words))
    cfd = ConditionalFreqDist(bigrams)
    
    current_word = random.choice(words)  # Changed from nltk.random.choice
    generated_words = [current_word]
    
    for _ in range(num_words - 1):
        if current_word in cfd:
            possible_words = list(cfd[current_word].keys())
            if possible_words:
                current_word = random.choice(possible_words)  # Changed from nltk.random.choice
                generated_words.append(current_word)
            else:
                current_word = random.choice(words)  # Changed from nltk.random.choice
                generated_words.append(current_word)
        else:
            current_word = random.choice(words)  # Changed from nltk.random.choice
            generated_words.append(current_word)
    
    return ' '.join(generated_words)

# Generate sample texts from different corpora
print("\n=== Random Text Generation ===")
print("\nFrom Brown Corpus:")
print(generate_random_text(brown))
print("\nFrom Reuters Corpus:")
print(generate_random_text(reuters))

# 6. Create and analyze your own corpus
def create_custom_corpus(text):
    """Example of creating and analyzing a custom corpus"""
    tokens = word_tokenize(text)
    freq_dist = FreqDist(tokens)
    
    print("\n=== Custom Corpus Analysis ===")
    print(f"Total words: {len(tokens)}")
    print(f"Unique words: {len(set(tokens))}")
    print("\nMost common words:")
    print(freq_dist.most_common(5))
    
    # Plot frequency distribution
    plt.figure(figsize=(10, 5))
    freq_dist.plot(30, cumulative=True)
    plt.title("Word Frequency Distribution in Custom Corpus")
    plt.tight_layout()
    plt.savefig('custom_corpus_freq.png')

# Example usage with sample text
sample_text = """
Natural Language Processing (NLP) is a subfield of artificial intelligence
that focuses on the interaction between computers and human language.
NLP tasks include text classification, machine translation, and sentiment analysis.
"""

create_custom_corpus(sample_text)

