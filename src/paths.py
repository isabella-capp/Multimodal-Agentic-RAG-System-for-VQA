import os

BASE_FOLDER = os.getenv("EVQA_DATA", "/work/cvcs2026/encyclopedic")

JSON_PATH = f"{BASE_FOLDER}/encyclopedic_test_subset.json"
KB_PATH = f"{BASE_FOLDER}/encyclopedic_kb_wiki.db"
IMG_INDEX_PATH = f"{BASE_FOLDER}/knn.index"
IMG_INDEX_JSON_PATH = f"{BASE_FOLDER}/knn.json"

TERM_DF_PATH = "outputs/retrieval/term_df.json"

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
RETRIEVER_DEVICE = "cuda"
EF_SEARCH = 256  # HNSW search depth; the ~16 default under-retrieves
