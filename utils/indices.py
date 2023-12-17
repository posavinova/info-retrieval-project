import json
import logging
import math
from typing import Dict
from functools import lru_cache

import gensim
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from constants import (
    WORD2VEC_INVERTED_INDEX_STORAGE, INVERTED_INDEX_STORAGE, BM25_INVERTED_INDEX_STORAGE, WORD2VEC_MODEL_PATH,
    FASTTEXT_MODEL_PATH, FASTTEXT_INVERTED_INDEX_STORAGE, SBERT_MODEL, SBERT_INVERTED_INDEX_STORAGE
)
from utils.preprocess import preprocess_text

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


def get_document_embedding(model, text):
    """
    Get the document embedding for a given text using a FastText or Word2Vec model.
    :param model: FastText or Word2Vec model
    :param text: List of tokens representing the document text
    :return: Document embedding as a numpy array
    """
    word_vectors = []

    for token in text:
        if token in model.wv:
            word_vector = model.wv[token]
            word_vectors.append(word_vector)

    if word_vectors:
        doc_embedding = np.mean(word_vectors, axis=0)
    else:
        doc_embedding = np.zeros(model.vector_size)  # Default to zero vector if no valid tokens

    return doc_embedding


def build_static_vectors_index(documents: Dict[str, str], model_type: str) -> None:
    logger.info(f"Building inverted index with {model_type} model.")

    if model_type == "word2vec":
        try:
            vectorizer = gensim.models.Word2Vec.load(str(WORD2VEC_MODEL_PATH))
        except FileNotFoundError:
            logger.error(f"Word2Vec model file not found at {WORD2VEC_MODEL_PATH}.")
            return
        storage = WORD2VEC_INVERTED_INDEX_STORAGE
    elif model_type == "fasttext":
        try:
            vectorizer = gensim.models.FastText.load(str(FASTTEXT_MODEL_PATH))
        except FileNotFoundError:
            logger.error(f"FastText model file not found at {FASTTEXT_MODEL_PATH}.")
            return
        storage = FASTTEXT_INVERTED_INDEX_STORAGE
    else:
        logger.error(f"Could not load {model_type} model.")
        return

    static_vectors_index = {}

    for doc_name, doc_text in documents.items():
        doc_embedding = get_document_embedding(model=vectorizer, text=doc_text)
        static_vectors_index[doc_name] = doc_embedding.tolist()

    with open(storage, mode="w") as file:
        json.dump(static_vectors_index, file)


def build_bm25_index(documents: Dict[str, str], k: float = 1.5, b: float = 0.75) -> None:
    logger.info("Building inverted index with BM-25.")
    bm25_index = {}

    doc_lengths = [len(doc.split()) for doc in documents.values()]
    avg_doc_length = sum(doc_lengths) / len(documents)
    for doc_name, doc_text in documents.items():
        processed_text = preprocess_text(doc_text)
        doc_length = len(processed_text)
        for word in processed_text:
            if word not in bm25_index:
                bm25_index[word] = {}
            if doc_name not in bm25_index[word]:
                bm25_index[word][doc_name] = 0
            tf = processed_text.count(word)
            idf = math.log((len(documents) - len(bm25_index[word]) + 0.5) / (len(bm25_index[word]) + 0.5))
            bm25 = (tf * (k + 1)) / (tf + k * (1 - b + b * (doc_length / avg_doc_length)))
            bm25_index[word][doc_name] = bm25 * idf

    with open(BM25_INVERTED_INDEX_STORAGE, mode="w") as file:
        json.dump(bm25_index, file)


def build_frequency_index(documents: Dict[str, str]) -> None:
    logger.info("Building inverted index with frequencies.")
    frequency_index = {}

    for doc_name, doc_text in documents.items():
        processed_text = preprocess_text(doc_text)
        unique_words = set(processed_text)
        for word in unique_words:
            if word not in frequency_index:
                frequency_index[word] = {}
            if doc_name not in frequency_index[word]:
                frequency_index[word][doc_name] = processed_text.count(word)

    with open(INVERTED_INDEX_STORAGE, mode="w") as file:
        json.dump(frequency_index, file)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


@lru_cache
def load_sbert():
    tokenizer = AutoTokenizer.from_pretrained(SBERT_MODEL)
    model = AutoModel.from_pretrained(SBERT_MODEL)
    return tokenizer, model


def build_sbert_index(documents: Dict[str, str]) -> None:
    logger.info(f"Building inverted index with sbert model.")

    sbert_inverted_index = {}

    tokenizer, model = load_sbert()

    for doc_name, doc_text in tqdm(documents.items()):
        encoded_input = tokenizer([doc_text], padding=True, truncation=True, max_length=24, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**encoded_input)
        doc_embedding = mean_pooling(model_output, encoded_input["attention_mask"])
        sbert_inverted_index[doc_name] = doc_embedding.tolist()

    with open(SBERT_INVERTED_INDEX_STORAGE, mode="w") as file:
        json.dump(sbert_inverted_index, file)
