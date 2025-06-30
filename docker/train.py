import argparse
from datetime import datetime, timedelta
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Subset
from torch.optim import AdamW


from src.common.dataset import KlinesDataset
from src.common.basic_model import BasicModel
from src.common.normalize import robust_zscore_norm
from src.common.random import set_random_seed
from src.train.train import train


def main(
    dataset_dir,
    target_symbols,
    seed,
    fold_number,
    period_start_date,
    period_train_len,
    period_val_len,
    input_length,
    prediction_length,
    hidden_dim,
    num_layers,
    learning_rate,
    dropout,
    weight_decay,
):
    job_start_time = datetime.now()

    dataset_dir = Path(dataset_dir)

    target_symbols = target_symbols.strip().split(",")
    start_date = datetime.fromisoformat(period_start_date) + timedelta(days=period_val_len * fold_number)

    torch.set_default_device('cuda')

    set_random_seed(seed)

    train_dataset_list = []
    val_dataset_list = []
    for symbol in target_symbols:
        dataset = KlinesDataset(
            dataset_dir / f"{symbol}.hdf5",
            input_length=input_length,
            y_prediction_length=prediction_length,
        )
        train_indices = dataset.oversampling(
            dataset.filter_by_period(
                start_date - timedelta(days=period_train_len),
                start_date,
                sample_rate=1.0,
            )
        )
        dataset.normalize(train_indices, robust_zscore_norm)
        train_dataset = Subset(
            dataset,
            train_indices,
        )
        val_dataset = Subset(
            dataset,
            dataset.filter_by_period(
                start_date,
                start_date + timedelta(days=period_val_len),
                sample_rate=1.0,
            )
        )
        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)

    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    params = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }
    model = BasicModel(
        input_dim=dataset.num_features,
        **params
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train(
        model,
        optimizer,
        train_dataset,
        val_dataset,
    )

    print(f"job duration={(datetime.now() - job_start_time).total_seconds()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, default="/opt/ml/input/data/train")

    parser.add_argument("--target-symbols")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold-number", type=int)
    parser.add_argument("--period-start-date")
    parser.add_argument("--period-train-len", type=int)
    parser.add_argument("--period-val-len", type=int)

    parser.add_argument("--input-length", type=int)
    parser.add_argument("--prediction-length", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--weight-decay", type=float)

    args = parser.parse_args()

    main(
        args.train,
        args.target_symbols,
        args.seed,
        args.fold_number,
        args.period_start_date,
        args.period_train_len,
        args.period_val_len,
        args.input_length,
        args.prediction_length,
        args.hidden_dim,
        args.num_layers,
        args.learning_rate,
        args.dropout,
        args.weight_decay,
    )
