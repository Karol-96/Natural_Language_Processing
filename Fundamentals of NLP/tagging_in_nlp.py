import nltk
from nltk import DefaultTagger, RegexpTagger, UnigramTagger, BigramTagger
from nltk.corpus import brown
from nltk.tag import untag
import random


# Download required NLTK data
nltk.download('brown',quiet=True)
nltk.download('averaged_perceptron_tagger',quiet=True)
nltk.download('universal_tagset',quiet=True)

def demonstrate_basic_tagging():
    """
    Demonstrates basic POS tagging concepts
    """
    # Add this line at the beginning of the function
    nltk.download('averaged_perceptron_tagger')
    
    print("\n=== Basic POS Tagging Demo ===")
    
    # Rest of the function remains the same
    text = "The quick brown fox jumps over the lazy dog"
    tokens = nltk.word_tokenize(text)
    
    tagged = nltk.pos_tag(tokens)
    print("\nBasic NLTK Tagger:")
    print(tagged)
    
    tagged_universal = nltk.pos_tag(tokens, tagset='universal')

def create_train_test_split(tagged_sents, split=0.8):
    """
    Creates training and testing splits from tagged sentences
    """
    split_size = int(len(tagged_sents) * split)
    train_sents = tagged_sents[:split_size]
    test_sents = tagged_sents[split_size:]
    return train_sents, test_sents

def evaluate_tagger(tagger, test_sents):
    """
    Evaluates tagger accuracy
    """
    gold_tokens = sum(len(sent) for sent in test_sents)
    predicted_tokens = 0
    correct_tokens = 0
    
    for sent in test_sents:
        untagged = untag(sent)
        predicted = tagger.tag(untagged)
        predicted_tokens += len(predicted)
        correct_tokens += sum(1 for ((w1, t1), (w2, t2)) in zip(sent, predicted) if t1 == t2)
    
    accuracy = correct_tokens / gold_tokens
    return accuracy

def demonstrate_default_tagger():
    """
    Demonstrates Default Tagger
    """
    print("\n=== Default Tagger Demo ===")
    
    # Create and evaluate default tagger
    default_tagger = DefaultTagger('NN')  # Tags everything as noun
    
    # Test on a simple sentence
    tokens = ['The', 'quick', 'brown', 'fox']
    tagged = default_tagger.tag(tokens)
    print(f"\nDefault Tagger results (everything tagged as NN):")
    print(tagged)
    
    # Evaluate on Brown corpus
    brown_tagged_sents = brown.tagged_sents(categories='news')[:1000]
    accuracy = evaluate_tagger(default_tagger, brown_tagged_sents)
    print(f"Default Tagger accuracy: {accuracy:.2%}")

def demonstrate_regexp_tagger():
    """
    Demonstrates Regular Expression Tagger
    """
    print("\n=== RegexpTagger Demo ===")
    
    # Define patterns
    patterns = [
        (r'.*ing$', 'VBG'),               # gerunds
        (r'.*ed$', 'VBD'),                # simple past
        (r'.*es$', 'VBZ'),                # 3rd singular present
        (r'.*ould$', 'MD'),               # modals
        (r'.*\'s$', 'NN$'),               # possessive nouns
        (r'.*s$', 'NNS'),                 # plural nouns
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'.*', 'NN')                     # nouns (default)
    ]
    
    regexp_tagger = RegexpTagger(patterns)
    
    # Test on sample text
    text = "The running dog chased cats and could bark"
    tokens = text.split()
    tagged = regexp_tagger.tag(tokens)
    print("\nRegexp Tagger results:")
    print(tagged)

def demonstrate_ngram_taggers():
    """
    Demonstrates N-gram Taggers (Unigram and Bigram)
    """
    print("\n=== N-gram Taggers Demo ===")
    
    # Prepare training and testing data
    brown_tagged_sents = brown.tagged_sents(categories='news')
    train_sents, test_sents = create_train_test_split(brown_tagged_sents)
    
    # Create and train Unigram tagger
    unigram_tagger = UnigramTagger(train_sents)
    unigram_accuracy = evaluate_tagger(unigram_tagger, test_sents)
    print(f"\nUnigram Tagger accuracy: {unigram_accuracy:.2%}")
    
    # Create and train Bigram tagger
    bigram_tagger = BigramTagger(train_sents)
    bigram_accuracy = evaluate_tagger(bigram_tagger, test_sents)
    print(f"Bigram Tagger accuracy: {bigram_accuracy:.2%}")

def demonstrate_backoff_tagger():
    """
    Demonstrates combining taggers using backoff
    """
    print("\n=== Backoff Tagger Demo ===")
    
    # Prepare data
    brown_tagged_sents = brown.tagged_sents(categories='news')
    train_sents, test_sents = create_train_test_split(brown_tagged_sents)
    
    # Create taggers with backoff chain
    default_tagger = DefaultTagger('NN')
    unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)
    bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)
    
    # Evaluate combined tagger
    accuracy = evaluate_tagger(bigram_tagger, test_sents)
    print(f"\nBackoff Tagger accuracy: {accuracy:.2%}")
    
    # Example usage
    text = "The quick brown fox jumps over the lazy dog"
    tokens = text.split()
    tagged = bigram_tagger.tag(tokens)
    print("\nBackoff Tagger results:")
    print(tagged)

def practical_example():
    """
    Shows a practical application of POS tagging
    """
    print("\n=== Practical Application: Extracting Noun Phrases ===")
    
    text = """
    The beautiful butterfly landed on a colorful flower.
    The young scientist conducted important research in the modern laboratory.
    """
    
    # Tokenize and tag
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    
    # Extract adjective-noun sequences
    noun_phrases = []
    for i in range(len(tagged)-1):
        if tagged[i][1].startswith('JJ') and tagged[i+1][1].startswith('NN'):
            noun_phrases.append(f"{tagged[i][0]} {tagged[i+1][0]}")
    
    print("\nExtracted adjective-noun phrases:")
    for phrase in noun_phrases:
        print(f"- {phrase}")

def main():
    """
    Main function to run all demonstrations
    """
    print("=== Part of Speech (POS) Tagging Demonstration ===")
    print("This program demonstrates various POS tagging techniques")
    
    demonstrate_basic_tagging()
    demonstrate_default_tagger()
    demonstrate_regexp_tagger()
    demonstrate_ngram_taggers()
    demonstrate_backoff_tagger()
    practical_example()

if __name__ == "__main__":
    main()