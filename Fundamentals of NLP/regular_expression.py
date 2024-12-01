import re
import unicodedata

def unicode_demo():
    """
    Demonstrates basic Unicode text processing
    """
    print("\n=== Unicode Text Processing ===")
    
    # Unicode string examples
    text = "Hello, 世界! 🌍"
    print(f"Original text: {text}")
    
    # Converting to different encodings
    utf8_bytes = text.encode('utf-8')
    print(f"UTF-8 encoded: {utf8_bytes}")
    print(f"Decoded back: {utf8_bytes.decode('utf-8')}")
    
    # Normalizing Unicode
    mixed_text = "café"  # can be represented in two ways
    nfkd_form = unicodedata.normalize('NFKD', mixed_text)
    print(f"\nNormalized form: {[ord(c) for c in nfkd_form]}")

def regex_pattern_demo():
    """
    Demonstrates various regex patterns and their uses
    """
    print("\n=== Regular Expression Pattern Matching ===")
    
    test_text = """
    Hello World! This is a test string with some patterns.
    Email: test@example.com
    Phone: 123-456-7890
    Date: 2024-03-20
    Website: https://www.example.com
    """
    
    # 1. Dollar sign ($) - End of word/line
    print("\n1. Finding words ending with 'ing':")
    ing_pattern = r'\w+ing$'
    ing_words = re.findall(ing_pattern, test_text, re.MULTILINE)
    print(f"Words ending in 'ing': {ing_words}")
    
    # 2. Dot symbol (.) - Any character
    print("\n2. Matching email pattern:")
    email_pattern = r'\S+@\S+\.\S+'
    emails = re.findall(email_pattern, test_text)
    print(f"Found emails: {emails}")
    
    # 3. Caret symbol (^) - Start of word/line
    print("\n3. Finding lines starting with 'Hello':")
    hello_pattern = r'^Hello'
    hello_matches = re.findall(hello_pattern, test_text, re.MULTILINE)
    print(f"Lines starting with 'Hello': {hello_matches}")
    
    # 4. Question mark (?) - Optional character
    print("\n4. Matching both 'color' and 'colour':")
    color_pattern = r'colou?r'
    test_str = "Both color and colour are correct"
    colors = re.findall(color_pattern, test_str)
    print(f"Found color variants: {colors}")
    
    # 5. Plus sign (+) - One or more instances
    print("\n5. Finding numbers with one or more digits:")
    number_pattern = r'\d+'
    numbers = re.findall(number_pattern, test_text)
    print(f"Found numbers: {numbers}")
    
    # 6. Asterisk (*) - Zero or more instances
    print("\n6. Finding words with optional 's':")
    plural_pattern = r'tests*'
    test_str = "test tests testing"
    plurals = re.findall(plural_pattern, test_str)
    print(f"