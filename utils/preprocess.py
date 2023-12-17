"""
Module for basic text documents preprocessing, such as tokenization, lemmatization and punctuation+stop-words removal
"""
import re
from typing import List

import nltk
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
stop_words = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()


def _tokenize_text(text: str) -> List[str]:
    text = re.sub(r"(?<=\s)[^\w\s]+|[^\w\s]+(?=\s)|\*+", "", text)
    tokens = word_tokenize(text, language="russian")
    return tokens


def _lemmatize_text(tokens: List[str]) -> List[str]:
    lemmatized_text = [morph.parse(token.lower())[0].normal_form for token in tokens]
    return lemmatized_text


def preprocess_text(text: str) -> List[str]:
    tokens = _tokenize_text(text)
    processed_text = _lemmatize_text(tokens)
    return processed_text
