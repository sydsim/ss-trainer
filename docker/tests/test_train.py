from datetime import datetime

import torch
import pytest

from torch.utils.data import ConcatDataset, Subset
from torch.optim import AdamW

from src.common.dataset import DummyKlinesDataset
from src.common.normalize import robust_zscore_norm
from src.train.train import train

from src.sm_model.models import CustomModel


def test_train():
    print(torch.get_default_device())

    dataset = DummyKlinesDataset(
        input_length=240,
        y_prediction_length=20,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        n=10000,
    )

    train_dataset_list = []
    val_dataset_list = []
    for _ in range(3):
        dataset = DummyKlinesDataset(
            input_length=240,
            y_prediction_length=20,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),
            n=10000,
        )
        train_indices = dataset.oversampling(
            dataset.filter_by_period(
                datetime(2023, 1, 1),
                datetime(2023, 7, 1),
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
                datetime(2023, 7, 1),
                datetime(2024, 1, 1),
                sample_rate=1.0,
            )
        )
        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)

    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    model = CustomModel(
        input_dim=dataset.num_features,
    )
    optimizer = AdamW(model.parameters())

    train(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        ckpt_dir=None,
        epochs=3,
    )


@pytest.mark.cpu
def test_train_cpu():
    test_train()


@pytest.mark.gpu
def test_forward_pass_gpu():
    torch.set_default_device('cuda')
    test_train()
