#In nlp, String tokenization is the process of breaking down text (like sentences or documents) into
#Smaller units called Tokens. These tokens can be words, characters, subwords or even phrases.


import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
import string 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 


# With these three lines:
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

text = "Hello! This is an example of Word Tokenization in nlp."

#Word Tokenization
word_tokens = word_tokenize(text)
print("Words Token: ", word_tokens)

#Sentence Tokenization
sent_token = sent_tokenize(text)
print("Sentence Token:", sent_token)

#Character Tokenization
def char_token(text):
    return list(text)

char_tokens = char_token(text)
print(char_tokens)


#Removing stop words and punctiuations
def clean_text():
    text = text.lower()

    #Tokenize the text
    tokens = word_tokenize(text)

    #get the english stop words
    stopwords = set(stopwords.words('english'))

    #Removing stop words and punctuation
    cleaned_tokens = [
        token                                    # Removed comma after tokens
        for token in tokens
        if (token not in stopwords and
            token not in string.punctuation)
    ]
    return cleaned_tokens

# Example usage
text = "Hello! This is an example sentence. It contains some stop words and punctuation marks..."

# Original text
print("Original text:", text)

# Clean the text
cleaned_text = clean_text(text)
print("\nCleaned text:", cleaned_text)