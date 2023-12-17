import argparse
import logging
import time
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from search import (
    get_search_results_frequencies, get_search_results_bm25,
    get_search_results_static_vectors, get_search_results_sbert
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

app = FastAPI()


class SearchQuery(BaseModel):
    query: str = Field(
        description="The search query", example="забыл пароль от лк"
    )
    search_type: str = Field(
        description="The type of search to perform (freq or bm25).", example="freq", enum=["freq", "bm25"]
    )
    top_n: Optional[int] = Field(
        default=None, description="Maximum number of most relevant results to return", example=10
    )


@app.get("/")
def get_info():
    return {"about": "Lexical search implementation"}


@app.post("/search")
def search(query_params: SearchQuery):
    query = query_params.query
    search_type = query_params.search_type
    top_n = query_params.top_n

    results, execution_time = base_search(query, search_type, top_n)
    if results:
        return {
            "execution_time": execution_time,
            "results":  [[rank, result[0], normalize_score(result[1])] for rank, result in enumerate(results, start=1)]
        }
    else:
        return {}


def create_parser():
    parser = argparse.ArgumentParser(description="Command Line Search Engine for Evotor")
    parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    parser.add_argument(
        "--search-type",
        choices=["freq", "bm25", "word2vec", "fasttext", "sbert"],
        default="bm25",
        help="Model or algorithm used for indexing and search"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top results to return"
    )
    return parser


def normalize_score(score):
    if isinstance(score, float):
        return score
    else:
        return score[0][0]


def base_search(query, search_type="bm25", top_n=None):
    start_time = time.time()
    if search_type == "freq":
        results = get_search_results_frequencies(query, search_type, top_n)
    elif search_type == "bm25":
        results = get_search_results_bm25(query, search_type, top_n)
    elif search_type in ["word2vec", "fasttext"]:
        results = get_search_results_static_vectors(query, search_type, top_n)
    elif search_type == "sbert":
        results = get_search_results_sbert(query, search_type, top_n)
    else:
        return [], None
    end_time = time.time()
    execution_time = end_time - start_time
    if results:
        return results, execution_time


def main():
    parser = create_parser()
    args = parser.parse_args()

    query = args.query
    search_type = args.search_type
    top_n = args.top_n

    results, execution_time = base_search(query, search_type, top_n)

    if not results:
        logger.info("No results found.")
    else:
        logger.info(f"Search executed within {execution_time}s")
        logger.info("Search Results:")
        for rank, (document, score) in enumerate(results, start=1):
            logger.info(f"{rank}. Document: {document}, Score: {normalize_score(score)}")


if __name__ == "__main__":
    main()
    # base_search("ЭЦП", search_type="sbert")
