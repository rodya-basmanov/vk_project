"""
Оценка модели: Recall@10 и MRR на dev-сплите MIRACL-ru.

Поддерживает два режима:
  --model pretrained   — оценка базовой (не дообученной) модели
  --model finetuned    — оценка fine-tuned модели

Запуск:
    python evaluate.py --model pretrained
    python evaluate.py --model finetuned
"""

import argparse
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from data_loader import get_eval_data


# ------------------------------------------------------------------
# Метрики
# ------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    """Доля найденных релевантных документов в top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / ранг первого релевантного документа."""
    for i, docid in enumerate(retrieved):
        if docid in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ------------------------------------------------------------------
# Основная оценка
# ------------------------------------------------------------------

def evaluate(model_type: str = "pretrained"):
    """
    Полный pipeline оценки:
      1. Загрузить модель
      2. Кодировать корпус и запросы
      3. Для каждого запроса — cosine similarity → top-k
      4. Подсчитать Recall@10 и MRR
    """

    # 1. Выбор модели
    if model_type == "finetuned":
        model_path = config.MODEL_DIR
        print(f"Загрузка fine-tuned модели: {model_path}")
    else:
        model_path = config.BASE_MODEL_NAME
        print(f"Загрузка baseline модели: {model_path}")

    model = SentenceTransformer(model_path, device=config.DEVICE)
    model.max_seq_length = config.MAX_SEQ_LENGTH

    # 2. Данные
    print("Загрузка eval-данных (dev)...")
    queries, corpus, relevant = get_eval_data("dev")

    # 3. Кодирование корпуса
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]

    print(f"Кодирование корпуса ({len(doc_texts)} документов)...")
    t0 = time.time()
    doc_embeddings = model.encode(
        doc_texts,
        batch_size=config.ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Время: {time.time() - t0:.1f} сек")

    # 4. Кодирование запросов
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"Кодирование запросов ({len(query_texts)})...")
    query_embeddings = model.encode(
        query_texts,
        batch_size=config.ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 5. Поиск top-k (cosine similarity через dot product нормализованных векторов)
    print("Поиск top-k...")
    recalls = []
    mrrs = []

    for i, qid in enumerate(tqdm(query_ids, desc="Оценка")):
        scores = query_embeddings[i] @ doc_embeddings.T
        top_indices = np.argsort(scores)[::-1][:config.TOP_K]
        retrieved = [doc_ids[idx] for idx in top_indices]

        rel = relevant.get(qid, set())
        recalls.append(recall_at_k(retrieved, rel, config.TOP_K))
        mrrs.append(reciprocal_rank(retrieved, rel))

    avg_recall = np.mean(recalls)
    avg_mrr = np.mean(mrrs)

    # 6. Вывод результатов
    print("\n" + "=" * 50)
    print(f"  Модель:      {model_type}")
    print(f"  Recall@{config.TOP_K}:  {avg_recall:.4f}")
    print(f"  MRR:         {avg_mrr:.4f}")
    print("=" * 50)

    target_recall = 0.75
    target_mrr = 0.60

    if avg_recall >= target_recall and avg_mrr >= target_mrr:
        print("Целевые метрики достигнуты")
    else:
        issues = []
        if avg_recall < target_recall:
            issues.append(f"Recall@10 ({avg_recall:.4f} < {target_recall})")
        if avg_mrr < target_mrr:
            issues.append(f"MRR ({avg_mrr:.4f} < {target_mrr})")
        print(f"Не достигнуто: {', '.join(issues)}")

    return avg_recall, avg_mrr


def main():
    parser = argparse.ArgumentParser(
        description="Оценка модели на MIRACL-ru dev")
    parser.add_argument(
        "--model",
        choices=["pretrained", "finetuned"],
        default="pretrained",
        help="Тип модели для оценки",
    )
    args = parser.parse_args()

    evaluate(args.model)


if __name__ == "__main__":
    main()
