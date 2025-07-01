import argparse
from pathlib import Path
from src.common.dataset import KlinesDataset
from datetime import datetime
import torch

from src.simulation.simulation import run_simulation


def main(
    model_dir,
    dataset_dir,
    orderbook_dir,
    result_dir,

    target_symbol,
    period_start_date,
    period_train_len,
    period_val_len,
    num_folds,
    num_seeds,

    input_length,
    prediction_length,
    hidden_dim,
    num_layers,

    initial_balance,

    order_threshold,
    signal_threshold,
    trade_lifecycle,

    alpha,
    beta,

    device="cuda",
):
    job_start_time = datetime.now()

    torch.set_default_device(device)

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    orderbook_dir = Path(orderbook_dir)
    result_dir = Path(result_dir)

    period_start_date = datetime.fromisoformat(period_start_date)

    dataset = KlinesDataset(
        dataset_dir / f"{target_symbol}.hdf5",
        input_length=input_length,
        y_prediction_length=prediction_length,
    )

    run_simulation(
        model_dir,
        dataset,
        orderbook_dir,
        result_dir,

        target_symbol,
        period_start_date,
        period_train_len,
        period_val_len,
        num_folds,
        num_seeds,

        hidden_dim,
        num_layers,
        initial_balance,
        order_threshold,
        signal_threshold,
        trade_lifecycle,

        alpha,
        beta,
    )

    print(f"job duration={(datetime.now() - job_start_time).total_seconds()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="/opt/ml/model")
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--orderbook", type=str, default="/opt/ml/input/data/orderbook")
    parser.add_argument("--result", type=str, default="/opt/ml/output/result")

    parser.add_argument("--target-symbol")
    parser.add_argument("--period-start-date")
    parser.add_argument("--period-train-len", type=int)
    parser.add_argument("--period-val-len", type=int)
    parser.add_argument("--num-folds", type=int)
    parser.add_argument("--num-seeds", type=int)

    parser.add_argument("--input-length", type=int)
    parser.add_argument("--prediction-length", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)

    parser.add_argument("--initial-balance", type=float, default=1e6)

    parser.add_argument("--order-threshold", type=float, default=0.9)
    parser.add_argument("--signal-threshold", type=float, default=0.01)
    parser.add_argument("--trade-lifecycle", type=int, default=20)

    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--beta", type=float, default=1)

    args = parser.parse_args()

    main(
        args.model,
        args.train,
        args.orderbook,
        args.result,

        args.target_symbol,
        args.period_start_date,
        args.period_train_len,
        args.period_val_len,
        args.num_folds,
        args.num_seeds,

        args.input_length,
        args.prediction_length,
        args.hidden_dim,
        args.num_layers,

        args.initial_balance,

        args.order_threshold,
        args.signal_threshold,
        args.trade_lifecycle,

        args.alpha,
        args.beta,
    )
