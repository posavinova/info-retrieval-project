import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List, Union

import torch
from gensim.models import FastText, Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from constants import (
    RAW_DATA_PATH, CLEAN_DATA_PATH, BM25_INVERTED_INDEX_STORAGE, INVERTED_INDEX_STORAGE, FASTTEXT_MODEL_PATH,
    FASTTEXT_INVERTED_INDEX_STORAGE, WORD2VEC_INVERTED_INDEX_STORAGE, WORD2VEC_MODEL_PATH, SBERT_INVERTED_INDEX_STORAGE
)
from utils.indices import (
    build_frequency_index, build_bm25_index, build_static_vectors_index,
    get_document_embedding, build_sbert_index, mean_pooling, load_sbert
)
from utils.preprocess import preprocess_text

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


def process_raw_data(raw_data_path: Path, clean_data_path: Path) -> None:
    logger.info("Starting preprocessing documents corpus for search.")
    try:
        clean_data_path.mkdir()
    except FileExistsError:
        shutil.rmtree(clean_data_path)
        clean_data_path.mkdir()

    for document in tqdm(list(raw_data_path.iterdir())):
        with open(document) as raw_file:
            raw_data = raw_file.read()
        clean_data = preprocess_text(raw_data)
        with open(clean_data_path / document.name, mode="w") as clean_file:
            clean_file.write(" ".join(clean_data))


def get_documents() -> Dict[str, str]:
    if not CLEAN_DATA_PATH.exists():
        process_raw_data(RAW_DATA_PATH, CLEAN_DATA_PATH)
    documents = {}
    for doc in CLEAN_DATA_PATH.iterdir():
        with open(doc) as file:
            documents[doc.stem] = file.read()
    return documents


def get_search_results_frequencies(query: str, search_type: str = "freq", top_n: Union[int, None] = None) -> List[tuple]:
    processed_query = preprocess_text(query)
    scores = defaultdict(float)

    if search_type == "freq":
        index_storage = INVERTED_INDEX_STORAGE
        build_index_func = build_frequency_index
    else:
        logger.info(f"An error occurred. There is no such type of search {search_type}.")
        return []

    if not index_storage.exists():
        build_index_func(get_documents())

    with open(index_storage) as file:
        index = json.load(file)

    for word in processed_query:
        if word in index:
            for doc_id, score in index[word].items():
                scores[doc_id] += score

    top_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None and isinstance(top_n, int):
        top_documents = top_documents[:top_n]

    return top_documents


def get_search_results_bm25(query: str, search_type: str = "bm25", top_n: Union[int, None] = None) -> List[tuple]:
    processed_query = preprocess_text(query)
    scores = defaultdict(float)

    if search_type == "bm25":
        index_storage = BM25_INVERTED_INDEX_STORAGE
        build_index_func = build_bm25_index
    else:
        logger.info(f"An error occurred. There is no such type of search {search_type}.")
        return []

    if not index_storage.exists():
        build_index_func(get_documents())

    with open(index_storage) as file:
        index = json.load(file)

    for word in processed_query:
        if word in index:
            for doc_id, score in index[word].items():
                scores[doc_id] += score

    top_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None and isinstance(top_n, int):
        top_documents = top_documents[:top_n]

    return top_documents


def get_search_results_static_vectors(query: str, search_type: str, top_n: Union[int, None] = None) -> List[tuple]:
    scores = defaultdict(float)

    if search_type == "fasttext":
        index_storage = FASTTEXT_INVERTED_INDEX_STORAGE
        model = FastText.load(str(FASTTEXT_MODEL_PATH))
    elif search_type == "word2vec":
        index_storage = WORD2VEC_INVERTED_INDEX_STORAGE
        model = Word2Vec.load(str(WORD2VEC_MODEL_PATH))
    else:
        logger.info(f"An error occurred. There is no such type of search {search_type}.")
        return []

    build_index_func = build_static_vectors_index
    if not index_storage.exists():
        build_index_func(get_documents(), search_type)

    with open(index_storage) as file:
        index = json.load(file)

    query_tokens = " ".join(preprocess_text(query))
    query_vector = get_document_embedding(model=model, text=query_tokens)

    for doc_id, doc_vector in index.items():
        similarity = cosine_similarity([query_vector], [doc_vector])
        scores[doc_id] = similarity

    top_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None and isinstance(top_n, int):
        top_documents = top_documents[:top_n]

    return top_documents


def get_search_results_sbert(query: str, search_type: str = "sbert", top_n: Union[int, None] = None) -> List[tuple]:

    if search_type == "sbert":
        index_storage = SBERT_INVERTED_INDEX_STORAGE
        build_index_func = build_sbert_index
    else:
        logger.info(f"An error occurred. There is no such type of search {search_type}.")
        return []

    if not index_storage.exists():
        build_index_func(get_documents())

    with open(index_storage) as file:
        index = json.load(file)

    tokenizer, model = load_sbert()
    encoded_input = tokenizer([query], padding=True, truncation=True, max_length=24, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_vector = mean_pooling(model_output, encoded_input["attention_mask"]).tolist()

    scores = defaultdict(float)

    for doc_id, doc_vector in index.items():
        similarity = cosine_similarity(query_vector, doc_vector)
        scores[doc_id] = similarity

    top_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None and isinstance(top_n, int):
        top_documents = top_documents[:top_n]

    return top_documents
