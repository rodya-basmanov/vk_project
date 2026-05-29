"""
Inference: ответ на вопрос пользователя с помощью обученной модели.

Режимы работы:
  1. CLI — интерактивный диалог
  2. Одиночный запрос:  python inference.py --query "Кто такой Юрий Гагарин?"

Модель кодирует корпус документов из MIRACL-dev и ищет ближайшие по смыслу.
"""

import argparse
import textwrap

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from data_loader import get_eval_data


class SemanticSearchEngine:
    """Движок семантического поиска по корпусу MIRACL."""

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = config.MODEL_DIR

        print(f"Загрузка модели: {model_path}")
        self.model = SentenceTransformer(model_path, device=config.DEVICE)
        self.model.max_seq_length = config.MAX_SEQ_LENGTH

        print("Загрузка корпуса документов...")
        queries, corpus, _ = get_eval_data("dev")
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did] for did in self.doc_ids]

        print(f"Кодирование {len(self.doc_texts)} документов...")
        self.doc_embeddings = self.model.encode(
            self.doc_texts,
            batch_size=config.ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print("Готово!\n")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Найти top_k наиболее релевантных документов.
        Возвращает список словарей {docid, text, score}.
        """
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores = q_emb[0] @ self.doc_embeddings.T
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "docid": self.doc_ids[idx],
                "text": self.doc_texts[idx],
                "score": float(scores[idx]),
            })
        return results


def print_results(query: str, results: list[dict]):
    """Красивый вывод результатов поиска."""
    print(f"\n{'─' * 60}")
    print(f"  Вопрос: {query}")
    print(f"{'─' * 60}")

    for i, r in enumerate(results, 1):
        text_wrapped = textwrap.fill(r["text"], width=70, initial_indent="    ", subsequent_indent="    ")
        print(f"\n  [{i}] Релевантность: {r['score']:.4f}")
        print(f"      DocID: {r['docid']}")
        print(text_wrapped)

    print(f"\n{'─' * 60}\n")


def interactive_mode(engine: SemanticSearchEngine, top_k: int):
    """Интерактивный CLI-режим."""
    print("=" * 60)
    print("  Голосовой помощник Маруся — Семантический поиск")
    print("  Введите вопрос или 'выход' для завершения")
    print("=" * 60)

    while True:
        try:
            query = input("\nВаш вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("выход", "exit", "q", "quit"):
            print("До свидания!")
            break

        results = engine.search(query, top_k=top_k)
        print_results(query, results)


def main():
    parser = argparse.ArgumentParser(description="Семантический поиск по MIRACL-ru")
    parser.add_argument("--query", type=str, default=None, help="Одиночный запрос")
    parser.add_argument("--top_k", type=int, default=5, help="Количество результатов")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Путь к модели (по умолчанию — finetuned из config)",
    )
    args = parser.parse_args()

    engine = SemanticSearchEngine(model_path=args.model)

    if args.query:
        results = engine.search(args.query, top_k=args.top_k)
        print_results(args.query, results)
    else:
        interactive_mode(engine, top_k=args.top_k)


if __name__ == "__main__":
    main()
