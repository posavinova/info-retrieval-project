import logging
import multiprocessing
from typing import List

from gensim.models import Word2Vec, FastText

from constants import CLEAN_DATA_PATH, FASTTEXT_MODEL_PATH, WORD2VEC_MODEL_PATH


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


def train_vectorizer(dataset: List[List[str]], model_type) -> None:
    """
    Trains and saves word2vec or fasttext model
    :param dataset: tokenized documents
    :param model_type: vectorizer model - fasttext or word2vec
    :return: None
    """
    if model_type == "word2vec":
        model = Word2Vec(
            dataset,
            vector_size=100,
            window=5,
            min_count=1,
            sg=0,
            hs=0,
            negative=5,
            workers=multiprocessing.cpu_count()
        )
        model.save(str(WORD2VEC_MODEL_PATH))
        logger.info(f"Trained and persisted model at {WORD2VEC_MODEL_PATH}")
        return
    elif model_type == "fasttext":
        model = FastText(
            dataset,
            vector_size=100,
            window=5,
            min_count=5,
            sg=0,
            hs=0,
            negative=5,
            workers=-1
        )
        model.save(str(FASTTEXT_MODEL_PATH))
        logger.info(f"Trained and persisted model at {FASTTEXT_MODEL_PATH}")
        return
    else:
        logger.info(f"Nothing to persist. Could not train model {model_type}")
        return


if __name__ == "__main__":
    data = []
    for file in CLEAN_DATA_PATH.iterdir():
        with open(file) as document:
            data.append(document.read().split())
    train_vectorizer(dataset=data, model_type="fasttext")
    train_vectorizer(dataset=data, model_type="word2vec")
