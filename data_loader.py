from datasets import load_dataset
from sentence_transformers import InputExample
from tqdm import tqdm

import config


# ------------------------------------------------------------------
# Загрузка сырых данных
# ------------------------------------------------------------------

def load_miracl_data(split: str = "train"):
    """
    Загрузить MIRACL
    split: 'train', 'dev' или 'testA' (testB нельзя, т.к нет меток для валидации)

    """
    ds = load_dataset(
        config.DATASET_NAME,
        config.DATASET_LANG,
        split=split,
        trust_remote_code=True,
    )
    return ds


# ------------------------------------------------------------------
# Подготовка данных для обучения (для TripletLoss)
# ------------------------------------------------------------------

def prepare_training_triplets(max_samples=None):
    """
    Return:
        список примеров с тройками (query, positive, negative)

    """
    ds = load_miracl_data("train")
    examples = []

    for item in tqdm(ds, desc="Подготовка тренировочных троек"):
        query = item["query"]
        positives = item.get("positive_passages", [])
        negatives = item.get("negative_passages", [])

        if not positives or not negatives:
            continue

        n_pairs = min(len(positives), len(negatives))
        for i in range(n_pairs):
            pos_text = positives[i]["title"] + ". " + positives[i]["text"]
            neg_text = negatives[i]["title"] + ". " + negatives[i]["text"]
            examples.append(InputExample(texts=[query, pos_text, neg_text]))

        if max_samples and len(examples) >= max_samples:
            break

    print(f"Подготовлено {len(examples)} тренировочных троек")
    return examples


# ------------------------------------------------------------------
# Подготовка данных для оценки
# ------------------------------------------------------------------

def get_eval_data(split: str = "dev"):
    """
    Return:
      queries   — dict {qid: query_text}
      corpus    — dict {docid: doc_text}
      relevant  — dict {qid: set(docid, ...)}   (ground-truth)
    """
    ds = load_miracl_data(split)

    queries = {}
    corpus = {}
    relevant = {}

    for item in tqdm(ds, desc=f"Загрузка eval-данных ({split})"):
        qid = str(item["query_id"])
        queries[qid] = item["query"]
        relevant[qid] = set()

        for p in item.get("positive_passages", []):
            docid = p["docid"]
            corpus[docid] = p["title"] + ". " + p["text"]
            relevant[qid].add(docid)

        for p in item.get("negative_passages", []):
            docid = p["docid"]
            corpus[docid] = p["title"] + ". " + p["text"]

    print(f"Queries: {len(queries)}, Corpus docs: {len(corpus)}")
    return queries, corpus, relevant
