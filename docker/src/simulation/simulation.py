from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.basic.basic_state import BasicState
from src.common.normalize import robust_zscore_norm, softmax
from src.simulation.evaluation import evaluate
from src.simulation.datapoint import DataPoint

from src.sm_model.models import CustomModel


def validation_test(model, dataloader):
    model.eval()

    data_points = []
    y_preds = []
    with torch.no_grad():
        for ts, x, b_p, p_l, p_s, t_l, t_s in dataloader:
            for _ts, _b_p, _t_l, _t_s in zip(
                ts.cpu().numpy(),
                b_p.cpu().numpy(),
                t_l.cpu().numpy(),
                t_s.cpu().numpy(),
            ):
                r = DataPoint(_ts, _b_p, _t_l, _t_s, len(data_points))
                data_points.append(r)
            y_pred = model(x)
            y_preds.append(y_pred.cpu().numpy())

    return data_points, np.concatenate(y_preds, axis=0)


def load_orderbook(orderbook_dir, date):
    d = np.load(orderbook_dir / f"{date.date().isoformat()}/book_snapshot_25.npz")
    d = dict(d)
    ob_ts = d["ts"]
    ob_ask_price = np.stack([d[f"ask_{i}_price"] for i in range(20)], axis=-1)
    ob_ask_amount = np.stack([d[f"ask_{i}_amount"] for i in range(20)], axis=-1)
    ob_bid_price = np.stack([d[f"bid_{i}_price"] for i in range(20)], axis=-1)
    ob_bid_amount = np.stack([d[f"bid_{i}_amount"] for i in range(20)], axis=-1)
    return (
        ob_ts,
        ob_ask_price,
        ob_ask_amount,
        ob_bid_price,
        ob_bid_amount,
    )


def run(
    state, y_preds_all, data_points, ob_path, ob_start_date,
    order_threshold, signal_threshold,
):
    data_points = sorted(data_points, key=lambda x: x.timestamp)

    orderbook_date = ob_start_date
    (
        ob_ts,
        ob_ask_price,
        ob_ask_amount,
        ob_bid_price,
        ob_bid_amount,
    ) = load_orderbook(ob_path, orderbook_date)
    ob_i = 0

    for data_point in data_points:
        timestamp = data_point.timestamp
        base_price = data_point.base_price
        threshold_long = data_point.threshold_long
        threshold_short = data_point.threshold_short
        index = data_point.index

        while (ob_i < len(ob_ts) and ob_ts[ob_i] < timestamp):
            state.update_current_price(
                ob_ts[ob_i],
                bid_price=ob_bid_price[ob_i],
                bid_amount=ob_bid_amount[ob_i],
                ask_price=ob_ask_price[ob_i],
                ask_amount=ob_ask_amount[ob_i],
            )
            ob_i += 1
        if ob_i == len(ob_ts):
            orderbook_date += timedelta(days=1)
            (
                ob_ts,
                ob_ask_price,
                ob_ask_amount,
                ob_bid_price,
                ob_bid_amount,
            ) = load_orderbook(ob_path, orderbook_date)
            ob_i = 0

        state.step(timestamp)

        y_pred = y_preds_all[index, ...]
        prob = np.median(y_pred, axis=-1)
        prob_long = prob[1]
        prob_short = prob[2]

        state.open_order(timestamp, base_price, prob_long, prob_short, order_threshold, threshold_long, threshold_short, signal_threshold)


def run_simulation(
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

    batch_size=1024,
):
    params = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    state = BasicState(initial_balance=initial_balance, trade_lifecycle=trade_lifecycle)

    for fold_number in range(num_folds):
        start_date = period_start_date + timedelta(days=period_val_len * fold_number)

        train_indices = dataset.oversampling(
            dataset.filter_by_period(
                start_date - timedelta(days=period_train_len),
                start_date,
                sample_rate=1.0,
            )
        )
        dataset.normalize(train_indices, robust_zscore_norm)
        test_dataset = Subset(
            dataset,
            dataset.filter_by_period(
                start_date + timedelta(days=period_val_len),
                start_date + timedelta(days=period_val_len * 2),
                sample_rate=1.0,
            )
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=torch.Generator(device='cuda'),
            num_workers=0
        )
        model_list = []
        for seed in range(num_seeds):
            model = CustomModel(
                input_dim=dataset.num_features,
                **params
            )
            checkpoint = torch.load(model_dir / f"{seed}/{fold_number}/latest.pth")
            model.load_state_dict(checkpoint["best_model"])
            model_list.append(model)

        y_preds_all = []
        for model in model_list:
            data_points, y_preds = validation_test(model, test_dataloader)
            y_preds_all.append(y_preds)

        y_preds_all = np.stack(y_preds_all, axis=-1)
        y_preds_all = softmax(y_preds_all)
        run(
            state, y_preds_all, data_points, orderbook_dir / target_symbol, start_date + timedelta(days=period_val_len),
            order_threshold, signal_threshold,
        )

    test_start_date = period_start_date + timedelta(days=period_val_len)
    test_end_date = period_start_date + timedelta(days=period_val_len * (num_folds + 1))

    state.close_position(
        test_end_date.timestamp() * 1e6,
        force_all=True,
    )

    df = pd.DataFrame(state.trade_history, columns=["side", "created", "closed", "position_volume", "balance"])
    df["v"] = df.balance / df.balance.shift(1, fill_value=1e6)
    df["equity"] = df.balance
    df["ts"] = df.closed
    df.to_csv(result_dir / f"trades_{target_symbol}.csv")

    df = df[df.position_volume == 0]
    ev = evaluate(df, test_start_date, test_end_date, period_val_len * num_folds)

    print(f"backtest:balance={state.balance}")
    print(f"backtest:sharpe={ev['Sharpe Ratio']}")
    print(f"backtest:return={ev['Total Return']}")
    print(f"backtest:mdd={ev['Max Drawdown']}")
