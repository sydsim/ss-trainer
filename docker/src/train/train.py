

from copy import deepcopy
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.sm_loss.losses import CustomLoss


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


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    ckpt_dir=Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")),
    batch_size=1024,
    epochs=20,
    max_age=3,
):
    if ckpt_dir and (ckpt_dir / "latest.pth").is_file():
        checkpoint = torch.load(ckpt_dir / "latest.pth")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_avg_loss = checkpoint["best_avg_loss"]
        best_avg_epoch = checkpoint["best_avg_epoch"]
        best_state_dict = checkpoint["best_model"]
    else:
        start_epoch = 0
        best_avg_loss = 1e9
        best_avg_epoch = -1
        best_state_dict = None

    loss_function = CustomLoss()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=torch.get_default_device()),
        num_workers=0
    )
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device=torch.get_default_device()),
        num_workers=0
    )

    for epoch in range(start_epoch + 1, epochs):
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

        if ckpt_dir:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_model": best_state_dict,
                "best_avg_loss": best_avg_loss,
                "best_avg_epoch": best_avg_epoch,
            }, os.path.join(ckpt_dir, "latest.pth"))

    print(f"validation:best_loss={best_avg_loss}")
