# Comparing different stemmers:

# Word            Porter          Lancaster       Snowball        
# ------------------------------------------------------------
# running         run             run             run
# fishing         fish            fish            fish
# arguing         argu            argu            argu
# flying          fli             fly             fli
# crying          cri             cry             cri
# happiness       happi           happy           happi
# joyful          joy             joy             joy
# beautiful       beauti          beauti          beauti
# wonderful       wonder          wond            wonder
# calculation     calcul          calcul          calcul
# computing       comput          comput          comput
# calculated      calcul          calcul          calcul
# computes        comput          comput          comput
# authorization   author          auth            author
# authorized      author          auth            author
# authorizing     author          auth            author
# historically    histor          hist            histor
# history         histori         hist            histori
# historical      histor          hist            histor

# Real-world sentence example:

# Original: The running dogs were happily playing in the beautiful garden while the flying birds were singing historically

# Porter Stemmer: the run dog were happili play in the beauti garden while the fli bird were sing histor

# Lancaster Stemmer: the run dog wer hap play in the beauti gard whil the fly bird wer sing hist

# Snowball Stemmer: the run dog were happili play in the beauti garden while the fli bird were sing history


from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
import nltk

#Downloading required NLTK data
nltk.download('punkt',quiet=True)

#Initialize all three stemmers
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer()

#Testing words to demonstrate the differences
test_words = [
        'running', 'fishing', 'arguing', 'flying', 'crying',    # ing forms
    'happiness', 'joyful', 'beautiful', 'wonderful',        # adjectives and nouns
    'calculation', 'computing', 'calculated', 'computes',   # technical terms
    'authorization', 'authorized', 'authorizing',           # ization forms
    'historically', 'history', 'historical',  
]

#Function to compare all the stemmers
def compare_stemmers(words):
    print("{:<15} {:<15} {:<15} {:<15}".format(
        "Word", "Porter", "Lancaster", "Snowball"))
    print("-" * 60)

    for word in words:
        print("{:<15} {:<15} {:<15} {:<15}".format(
            word,
            porter.stem(word),
            lancaster.stem(word),
            snowball.stem(word)
        ))


#compare the stemmers
print('Comparing differenct stemmers:\n')
compare_stemmers(test_words)

#example with a sentence
sentence = "The running dogs were happily playing in the beautiful garden while the flying birds were singing historically"

# Tokenize the sentence
words = nltk.word_tokenize(sentence)

print("\nReal-world sentence example:\n")
print("Original:", sentence)

# Process with each stemmer
print("\nPorter Stemmer:", ' '.join([porter.stem(word) for word in words]))
print("Lancaster Stemmer:", ' '.join([lancaster.stem(word) for word in words]))
print("Snowball Stemmer:", ' '.join([snowball.stem(word) for word in words]))
