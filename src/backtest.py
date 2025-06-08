from copy import deepcopy
from pathlib import Path
from datetime import datetime
import random
import sys
import numpy as np
import torch
from tqdm import tqdm
from epitaph.datasets.klines_dataset import KlinesDataset
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW
from pytorch_optimizer import PCGrad
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from epitaph.losses.signal_loss import SignalLoss
from epitaph.losses.loss import UnifiedLoss, UnifiedProfitLoss
from epitaph.backtest.simulate import validation_test, model_analysis
from epitaph.models.mde_split import MDNSplitNormCopulaModel
from epitaph.models.signal import SignalModel
from epitaph.models.threshold import ThresholdModel
from epitaph.models.tft import TFTModel



def split_dataset(dataset):
    train_dataset = Subset(
        dataset,
        dataset.oversampling(
            dataset.filter_by_period(
                datetime(2021, 1, 1),
                datetime(2024, 1, 1),
                sample_rate=1.0,
            )
        )
    )
    valid_dataset = Subset(
        dataset,
        dataset.filter_by_period(
            datetime(2024, 1, 1),
            datetime(2024, 7, 1),
            sample_rate=1.0,
        )
    )
    test_dataset = Subset(
        dataset,
        dataset.filter_by_period(
            datetime(2024, 7, 1),
            datetime(2025, 1, 1),
            sample_rate=1.0,
        )
    )
    return train_dataset, valid_dataset, test_dataset


def train_one_epoch(model, dataloader, loss_function, optimizer):
    total_loss = 0
    step = 0
    for ts, x, b_p, p_l, p_s, t_l, t_s in tqdm(dataloader):
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

def train_cycle(symbol, t, epochs, model, train_dataloader, valid_dataloader, loss_function, optimizer, max_age):

    best_avg_loss = 1e9
    best_avg_epoch = -1
    for epoch in range(epochs):
        total_loss = train_one_epoch(model, train_dataloader, loss_function, optimizer)

        avg_loss = total_loss / len(train_dataloader.dataset)

        v_loss, _ = validation_test(
            model, valid_dataloader, loss_function
        )

        print(
            t, epoch, 
            "loss:", avg_loss, 
            "\nvalid loss:", v_loss, 
        )

        if best_avg_loss > v_loss:
            best_avg_loss = v_loss
            best_avg_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
        elif best_avg_epoch < epoch - max_age:
            break
        
        print("Best Vloss:", best_avg_loss)
        torch.save(model.state_dict(), f"results/snapshots/{symbol}_{t}_epoch-{epoch}.pth")
        torch.save(best_state_dict, f"results/snapshots/{symbol}_{t}.pth")


def train(symbol, model, train_dataset, valid_dataset, test_dataset, t, weight_decay):
    loss_function = UnifiedProfitLoss()

    if len(train_dataset) > 0 and len(valid_dataset) > 0:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1024,
            shuffle=True,
            generator=torch.Generator(device='cuda'),
            num_workers=0
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1024,
            shuffle=False,
            generator=torch.Generator(device='cuda'),
            num_workers=0
        )
        learning_rate = 1e-3
        epochs = 20
        max_age = 5

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_cycle(symbol, t, epochs, model, train_dataloader, valid_dataloader, loss_function, optimizer, max_age)

        # avg_loss, avg_dist_loss, avg_prob_loss, result = validation_test(
        #     model, valid_dataloader, loss_function
        # )
        # count_d, balance, df, sharpe_ratio, mdd, ldd, ts = model_analysis(
        #     result, loss_function, order_threshold=0.6,
        # )
        # print(avg_loss, avg_dist_loss, avg_prob_loss, balance, sharpe_ratio, mdd, ldd, count_d)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(symbol, dataset_date, seed):
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float64)

    base_path = Path("/home/nayuta/dataset/epitaph-2/")

    dataset = KlinesDataset(
        data_path = base_path / f"{symbol}-{dataset_date}.hdf5",
        y_prediction_length = 20,
        input_length = 240,
        selected_columns=[
            0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 
        ]
    )

    train_dataset, valid_dataset, test_dataset = split_dataset(dataset)

    params = {
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.4,
        "weight_decay": 1e-2,
    }
    weight_decay = params.pop("weight_decay")
    set_seed(seed)
    model = ThresholdModel(
        input_dim = dataset.num_features,
        bidirectional = False,
        **params
    )
    train(symbol, model, train_dataset, valid_dataset, test_dataset, seed, weight_decay=weight_decay)


"""
ADAUSDT-20250314.hdf5
BTCUSDT-20250305.hdf5 10
ETHUSDT-20250305.hdf5
SOLUSDT-20250309.hdf5
SUIUSDT-20250301.hdf5
XRPUSDT-20250309.hdf5
"""
if __name__ == "__main__":
    symbol = sys.argv[1]
    date = sys.argv[2]
    seed = int(sys.argv[3])
    main(
        symbol,
        date,
        seed
    )