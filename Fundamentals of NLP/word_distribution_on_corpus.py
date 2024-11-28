import nltk
from nltk.corpus import webtext, brown, reuters, inaugural
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import ssl
import numpy as np
from nltk.probability import FreqDist, ConditionalFreqDist

#ssl certificationm
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('webtext', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('reuters', quiet=True)
nltk.download('inaugural', quiet=True)
nltk.download('punkt')
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)  #


#Genre based analysis
# Download required NLTK data
nltk.download('webtext', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('reuters', quiet=True)
nltk.download('inaugural', quiet=True)
nltk.download('punkt', quiet=True)

def explore_corpus(corpus, title):
    """Helper function to explore basic corpus statistics"""
    print(f'\n====={title}=====')
    print(f'Number of words: {len(corpus.words())}')
    print(f"Number of unique words: {len(set(corpus.words()))}")
    print(f"First 5 words: {corpus.words()[:5]}")
    if hasattr(corpus,'categories'):
        print(f"Categories: {corpus.categories()[:5]}")

def analyze_modal_verbs_by_genre():
    """Analyze frequency of modal verbs across different genres"""
    # Define modal verbs to analyze
    modal_verbs = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
    
    # Select genres to analyze
    genres = ['news', 'religion', 'hobbies', 'fiction', 'science_fiction']
    
    # Create ConditionalFreqDist
    genre_modals = ConditionalFreqDist(
        (genre, word.lower())
        for genre in genres
        for word in brown.words(categories=genre)
        if word.lower() in modal_verbs
    )
    
    # Print tabulated results
    print("\n=== Modal Verb Usage Across Genres ===")
    genre_modals.tabulate(conditions=genres, samples=modal_verbs)
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Calculate percentages for better comparison
    genre_percentages = {}
    for genre in genres:
        total_words = len(brown.words(categories=genre))
        genre_percentages[genre] = {
            modal: (genre_modals[genre][modal] / total_words) * 100
            for modal in modal_verbs
        }
    
    # Plot grouped bar chart
    x = np.arange(len(modal_verbs))
    width = 0.15
    multiplier = 0
    
    for genre in genres:
        percentages = [genre_percentages[genre][modal] for modal in modal_verbs]
        plt.bar(x + width * multiplier, percentages, width, label=genre)
        multiplier += 1
    
    plt.xlabel('Modal Verbs')
    plt.ylabel('Frequency (%)')
    plt.title('Modal Verb Usage Across Different Genres')
    plt.xticks(x + width * 2, modal_verbs, rotation=45)
    plt.legend(title='Genres')
    plt.tight_layout()
    plt.savefig('modal_verb_analysis.png')

#Web and Chat Text Analysis
print("\n === Web and Chat Text Analysis ====")
files = webtext.fileids()
print(f'Available web text files: {files}')

#Analyze the chat message
chat = webtext.words('singles.txt')
chat_freq = FreqDist(word.lower() for word in chat)
print(f"\n Most common words in chat:")
print(chat_freq.most_common(10))

#Brown Corpus Analysis
print("\n==== Brown Corpus Analysis =====")
explore_corpus(brown, "Brown Corpus")

#Genre based analysis
genres = ['news','romance','science_fiction']
genre_words = [(genre,word)
               for genre in genres
               for word in brown.words(categories=genre)]

genre_dist = ConditionalFreqDist((genre,word.lower())
                                for genre,word in genre_words)


#Plot word frequencies across genres
plt.figure(figsize=(12,6))
#Plot word frequencies across genres
plt.figure(figsize=(12,6))
words_to_compare = ['the','love','space']

# Create bar plot
x = np.arange(len(words_to_compare))
width = 0.25
multiplier = 0

for genre in genres:
    frequencies = [genre_dist[genre][word] for word in words_to_compare]
    plt.bar(x + width * multiplier, frequencies, width, label=genre)
    multiplier += 1

plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title("Word Frequencies Across Genres")
plt.xticks(x + width, words_to_compare)
plt.legend()
plt.tight_layout()
plt.savefig('genre_frequencies.png')

# Reuters Corpus Analysis
print("\n=== Reuters Corpus Analysis ===")
explore_corpus(reuters, "Reuters Corpus")

# Analyze topics
topics = reuters.categories()
print(f"\nSample topics: {topics[:10]}")

def show_topic_stats(topic):
    files = reuters.fileids(categories=[topic])
    words = reuters.words(fileids=files)
    return len(files), len(words)

print("\nTopic Statistics:")
for topic in topics[:5]:
    files, words = show_topic_stats(topic)
    print(f"{topic}: {files} articles, {words} words")

# Inaugural Address Analysis
print("\n=== Inaugural Address Analysis ===")
explore_corpus(inaugural, "Inaugural Address Corpus")

def analyze_inaugural_timeline():
    cfd = ConditionalFreqDist(
        (target_year, word.lower())
        for target_year in inaugural.fileids()
        for word in inaugural.words(target_year)
    )
    
    words_of_interest = ['america', 'democracy', 'freedom', 'war']
    years = [f for f in inaugural.fileids()][-5:]  # Last 5 addresses
    
    print("\nWord usage in recent inaugural addresses:")
    cfd.tabulate(conditions=years, samples=words_of_interest)

analyze_inaugural_timeline()

def generate_random_text(corpus, num_words=50):
    """Generate random text using bigrams from the corpus"""
    import random
    
    words = corpus.words()
    bigrams = list(nltk.bigrams(words))
    cfd = ConditionalFreqDist(bigrams)
    
    current_word = random.choice(words)
    generated_words = [current_word]
    
    for _ in range(num_words - 1):
        if current_word in cfd:
            possible_words = list(cfd[current_word].keys())
            if possible_words:
                current_word = random.choice(possible_words)
                generated_words.append(current_word)
            else:
                current_word = random.choice(words)
                generated_words.append(current_word)
        else:
            current_word = random.choice(words)
            generated_words.append(current_word)
    
    return ' '.join(generated_words)

# Generate sample texts
print("\n=== Random Text Generation ===")
print("\nFrom Brown Corpus:")
print(generate_random_text(brown))
print("\nFrom Reuters Corpus:")
print(generate_random_text(reuters))

def create_custom_corpus(text):
    """Example of creating and analyzing a custom corpus"""
    tokens = word_tokenize(text)
    freq_dist = FreqDist(tokens)
    
    print("\n=== Custom Corpus Analysis ===")
    print(f"Total words: {len(tokens)}")
    print(f"Unique words: {len(set(tokens))}")
    print("\nMost common words:")
    print(freq_dist.most_common(5))
    
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

# Analyze modal verbs across genres
analyze_modal_verbs_by_genre()