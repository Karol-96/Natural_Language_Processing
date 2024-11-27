#Lemmatization is a text normalization technique that reduces words to their base or dictionary form
#(Known as lemma). Unlike stemming, which simply chops off words endings, lemmatization considers the context
#and part of speeech to apply the correct transformation rules.


# Basic Lemmatization Examples:
# running      -> running     
# flies        -> fly         
# driven       -> driven      
# better       -> good        
# children     -> child       

# Lemmatization with POS Tags:
# better       (a) -> good        
# better       (v) -> better      
# running      (v) -> run         
# running      (n) -> running     

# Real-world Sentence Examples:

# Original : The children were playing in the parks
# Lemmatized: The child be play in the park

# Original : The cats are running and catching mice
# Lemmatized: The cat be run and catch mouse

# Original : She was better at swimming than running
# Lemmatized: She be good at swim than run

# Original : The leaves are falling from the trees
# Lemmatized: The leaf be fall from the tree

# Comparison with Stemming:
# Word         Stem         Lemma       
# ------------------------------------
# caring       care         caring      
# cars         car          car         
# babies       babi         baby        
# are          are          be          
# better       better       good        
# worse        wors         worse       
# running      run          run         
# flies        fli          fly    


import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk import pos_tag 

#Download all teh required data
nltk.download('averaged_perceptron_tagger',quiet=True)
nltk.download('wordnet',quiet=True)
nltk.download('punkt',quiet=True)

def get_wordnet_pos(word):
    "Map POS tag to first character lemmatize() accepts"
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {

        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()


words = ['running', 'flies', 'driven', 'better', 'children']
for word in words:
    print(f"{word:12} -> {lemmatizer.lemmatize(word):12}")


# Example 2: Lemmatization with POS tags
print("\nLemmatization with POS Tags:")
word_pos_examples = [
    ('better', 'a'),    # adjective
    ('better', 'v'),    # verb
    ('running', 'v'),   # verb
    ('running', 'n'),   # noun
]
for word, pos in word_pos_examples:
    lemma = lemmatizer.lemmatize(word, pos=pos)
    print(f"{word:12} ({pos}) -> {lemma:12}")

# Example 3: Real-world sentence lemmatization
def lemmatize_sentence(sentence):
    # Tokenize the sentence
    words = nltk.word_tokenize(sentence)
    
    # Lemmatize each word with proper POS tag
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word))
        for word in words
    ]
    
    return ' '.join(lemmatized_words)

# Test sentences
sentences = [
    "The children were playing in the parks",
    "The cats are running and catching mice",
    "She was better at swimming than running",
    "The leaves are falling from the trees"
]