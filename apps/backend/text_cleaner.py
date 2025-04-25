from typing import List

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


try:
    stops = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stops = set(stopwords.words("english"))

tokenizer = RegexpTokenizer(r"\w+")
stemmer = PorterStemmer()


def clean(text: str) -> List[str]:
    """
    Preprocess text to make it more amenable to BM25.
    """
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stops]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens
