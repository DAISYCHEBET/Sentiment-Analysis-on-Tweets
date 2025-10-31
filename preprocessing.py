# preprocess.py
"""
Preprocessing utilities for the Twitter sentiment project.
Put this file in the same folder as your Streamlit `app.py` and model files.
Functions provided:
 - clean_text(text)
 - expand_contractions(text)
 - replace_slang(text)
 - tokenize_and_remove_stopwords(text)
 - lemmatize_with_pos(tokens)
 - preprocess_new_text(text)  # full pipeline returning a cleaned string

This file will attempt to download required NLTK resources on first run.
"""

import re
import string
import contractions
import emoji
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# Ensure required NLTK data packages are available (safe to call repeatedly)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


# --- slang dictionary (extend as needed) ---
slang_dict = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "idk": "i do not know",
    "lol": "laughing",
    "omg": "oh my god",
    "btw": "by the way",
    "brb": "be right back",
    "im": "i am",
    "tho": "though",
    "thx": "thanks",
    "pls": "please",
    "plz": "please"
}

# Initialize stopwords & lemmatizer
STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Helper: map POS tag to WordNet POS
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# 1) Basic cleaning
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # remove @mentions
    text = re.sub(r'@\w+', '', text)
    # remove rt token
    text = re.sub(r'\brt\b', '', text)
    # keep hashtag words but drop the '#' symbol
    text = re.sub(r'#', '', text)
    # demojize (convert emoji to text like :smiling_face: ) so they are tokenizable
    try:
        text = emoji.demojize(text, delimiters=(" ", " "))
    except Exception:
        pass
    # remove punctuation (after demojize so colons from demojize are removed)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# 2) Expand contractions
def expand_contractions(text: str) -> str:
    try:
        return contractions.fix(text)
    except Exception:
        return text


# 3) Replace slang / common shorthand
def replace_slang(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""
    words = text.split()
    new_words = [slang_dict.get(w, w) for w in words]
    return " ".join(new_words)


# 4) Tokenize and remove stopwords
def tokenize_and_remove_stopwords(text: str):
    if not isinstance(text, str) or text.strip() == "":
        return []
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in STOP_WORDS]
    return tokens


# 5) Lemmatize tokens with POS tagging
def lemmatize_with_pos(tokens):
    if not tokens:
        return []
    pos_tags = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return lemmas


# 6) Full preprocessing pipeline that returns a cleaned string (suitable for tokenizer)
def preprocess_new_text(text: str) -> str:
    """Run the full preprocessing pipeline and return a single cleaned string.
    The output is suitable for passing into the tokenizer (tokenizer.texts_to_sequences).
    """
    text = clean_text(text)
    text = expand_contractions(text)
    text = replace_slang(text)
    tokens = tokenize_and_remove_stopwords(text)
    lemmas = lemmatize_with_pos(tokens)
    return " ".join(lemmas)


# Small convenience function: preprocess multiple texts
def preprocess_texts(texts):
    return [preprocess_new_text(t) for t in texts]


# If you run this file directly, quick smoke test
if __name__ == '__main__':
    sample = "OMG I love this! LOL :) Visit https://example.com #fun @friend"
    print('Original:', sample)
    print('Preprocessed:', preprocess_new_text(sample))
