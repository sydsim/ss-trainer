import argparse
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.optim import AdamW


from datasets.klines_dataset import KlinesDataset
from models.basic_model import BasicModel
from losses.basic_loss import BasicLoss



def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, loss_function, optimizer):
    total_loss = 0
    step = 0
    for ts, x, b_p, p_l, p_s, t_l, t_s in dataloader:
        model.train()
        y_pred = model(x)
        
        loss = loss_function.get_loss(
            y_pred, b_p, p_l, p_s, t_l, t_s
        )
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        step += 1
    return total_loss

def validation_test(model, dataloader, loss_function):
    model.eval()
    total_loss = 0
    
    result = []
    with torch.no_grad():
        for ts, x, b_p, p_l, p_s, t_l, t_s in dataloader:
            r = [
                ts.cpu().numpy(),
                b_p.cpu().numpy(),
                p_l.cpu().numpy(),
                p_s.cpu().numpy(),
                t_l.cpu().numpy(),
                t_s.cpu().numpy(),
            ]
            
            y_pred = model(x)
            loss = loss_function.get_loss(
                y_pred, b_p, p_l, p_s, t_l, t_s
            )

            total_loss += loss.item()
            result.append(r + [y_pred.cpu().numpy()])

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, result

def main(
    model_dir,
    dataset_dir,
    seed,
    target_symbols,
    period_start_date,
    period_train_len,
    period_val_len,
    input_length,
    prediction_length,
    learning_rate,
    dropout,
    weight_decay,
    batch_size=1024,
    epochs=20,
    max_age=5,
):
    dataset_dir = Path(dataset_dir)
    model_dir = Path(model_dir)

    target_symbols = target_symbols.strip().split(",")
    start_date = datetime.fromisoformat(period_start_date)

    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float64)

    set_seed(seed)

    train_dataset_list = []
    val_dataset_list = []
    for symbol in target_symbols:
        dataset = KlinesDataset(
            dataset_dir / f"{symbol}.hdf5",
            input_length=input_length,
            y_prediction_length=prediction_length,
        )

        train_dataset = Subset(
            dataset,
            dataset.oversampling(
                dataset.filter_by_period(
                    start_date,
                    start_date + timedelta(days=period_train_len),
                    sample_rate=1.0,
                )
            )
        )
        val_dataset = Subset(
            dataset,
            dataset.filter_by_period(
                start_date + timedelta(days=period_train_len),
                start_date + timedelta(days=period_train_len + period_val_len),
                sample_rate=1.0,
            )
        )
        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)
    
    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    params = {
        "hidden_dim": 32,
        "num_layers": 2,
    }
    model = BasicModel(
        input_dim = dataset.num_features,
        **params
    )

    loss_function = BasicLoss()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device='cuda'),
        num_workers=0
    )
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device='cuda'),
        num_workers=0
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_avg_loss = 1e9
    best_avg_epoch = -1
    code = f"lr:{learning_rate}_do:{dropout}_wd:{weight_decay}"
    for epoch in range(epochs):
        total_loss = train_one_epoch(model, train_dataloader, loss_function, optimizer)

        avg_loss = total_loss / len(train_dataloader.dataset)

        v_loss, _ = validation_test(
            model, valid_dataloader, loss_function
        )

        print(f"train:loss={avg_loss}")
        print(f"validation:loss={v_loss}")

        if best_avg_loss > v_loss:
            best_avg_loss = v_loss
            best_avg_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
        elif best_avg_epoch < epoch - max_age:
            break
        
        torch.save(model.state_dict(), model_dir / f"{code}_epoch:{epoch}.pth")
        torch.save(best_state_dict, model_dir / f"{code}_best.pth")

    print(f"validation:best_loss={best_avg_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, default="/opt/ml/model")
    parser.add_argument('--train', type=str, default="/opt/ml/input/data/train")

    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-symbols")
    parser.add_argument("--period-start-date")
    parser.add_argument("--period-train-len", type=int)
    parser.add_argument("--period-val-len", type=int)

    parser.add_argument("--input-length", type=int)
    parser.add_argument("--prediction-length", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--weight-decay", type=float)

    args = parser.parse_args()
    
    main(
        args.model_dir,
        args.train,
        args.seed,
        args.target_symbols,
        args.period_start_date,
        args.period_train_len,
        args.period_val_len,
        args.input_length,
        args.prediction_length,
        args.learning_rate,
        args.dropout,
        args.weight_decay,
    )
