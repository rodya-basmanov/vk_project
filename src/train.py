import argparse
import logging
import os

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader

import config
from data_loader import load_miracl_data, prepare_training_triplets

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_dev_evaluator(sample_size=500):
    ds = load_miracl_data("dev")

    anchors, positives, negatives = [], [], []

    for item in ds:
        pos_list = item.get("positive_passages", [])
        neg_list = item.get("negative_passages", [])

        if not pos_list or not neg_list:
            continue

        anchors.append(item["query"])
        # Конкатенация название + текст
        positives.append(f"{pos_list[0]['title']}. {pos_list[0]['text']}")
        negatives.append(f"{neg_list[0]['title']}. {neg_list[0]['text']}")

        if len(anchors) >= sample_size:
            break

    return TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name="miracl-ru-dev",
        batch_size=config.EVAL_BATCH_SIZE,
    )


def train(epochs: int, batch_size: int, lr: float):
    logger.info(f"Init model: {config.BASE_MODEL_NAME}")
    model = SentenceTransformer(config.BASE_MODEL_NAME, device=config.DEVICE)
    model.max_seq_length = config.MAX_SEQ_LENGTH

    # Подготовка данных
    logger.info("Preparing training triplets...")
    train_examples = prepare_training_triplets()
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = get_dev_evaluator()
    warmup_steps = int(len(train_dataloader) * epochs * config.WARMUP_RATIO)

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    logger.info(
        f"Start training: epochs={epochs}, batch={batch_size}, lr={lr}, warmup={warmup_steps}"
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=config.MODEL_DIR,
        evaluation_steps=len(train_dataloader) // 2,
        save_best_model=True,
        show_progress_bar=True,
    )

    logger.info(f"Training complete. Model saved to: {config.MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ST on MIRACL-ru")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,
                        default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)

    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr)
