from pathlib import Path

CWD = Path.cwd()

RAW_DATA_PATH = Path(CWD / "raw_data")
CLEAN_DATA_PATH = Path(CWD / "clean_data")

INDICES_FOLDER = "indices"
MODELS_FOLDER = "models"

INVERTED_INDEX = "inverted_index.json"
SBERT_INVERTED_INDEX = "sbert_inverted_index.json"
WORD2VEC_INVERTED_INDEX = "word2vec_inverted_index.json"
FASTTEXT_INVERTED_INDEX = "fasttext_inverted_index.json"
BM25_INVERTED_INDEX = "BM25_inverted_index.json"

SBERT_MODEL = "ai-forever/sbert_large_nlu_ru"
WORD2VEC_MODEL = "word2vec_model.bin"
FASTTEXT_MODEL = "fasttext_model.bin"

WORD2VEC_MODEL_PATH = Path(CWD / MODELS_FOLDER / WORD2VEC_MODEL)
FASTTEXT_MODEL_PATH = Path(CWD / MODELS_FOLDER / FASTTEXT_MODEL)

SBERT_INVERTED_INDEX_STORAGE = Path(CWD / INDICES_FOLDER / SBERT_INVERTED_INDEX)
WORD2VEC_INVERTED_INDEX_STORAGE = Path(CWD / INDICES_FOLDER / WORD2VEC_INVERTED_INDEX)
FASTTEXT_INVERTED_INDEX_STORAGE = Path(CWD / INDICES_FOLDER / FASTTEXT_INVERTED_INDEX)
BM25_INVERTED_INDEX_STORAGE = Path(CWD / INDICES_FOLDER / BM25_INVERTED_INDEX)
INVERTED_INDEX_STORAGE = Path(CWD / INDICES_FOLDER / INVERTED_INDEX)
