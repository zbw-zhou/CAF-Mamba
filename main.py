"""
This is the main file to train and test the CAF-Mamba model for depression detection.
Notes:
    1. The model is trained and tested on both DVLOG and LMVD datasets.
    2. The CafMamba model is used to process the multimodal features of the LMVD dataset.
    3. The CafMambaBimodal model is used to process the bimodal features (acoustic features and facial landmarks) of both DVLOG and LMVD datasets.
Usage:
    python main.py --train --if_wandb --epochs 80 --batch_size 16 --learning_rate 1e-4 --model CafMamba --dataset lmvd --gpu 0 --scheduler
    python main.py --train --if_wandb --epochs 80 --batch_size 16 --learning_rate 1e-4 --model CafMambaBimodal --dataset lmvd --gpu 0 --scheduler
    python main.py --train --if_wandb --epochs 80 --batch_size 16 --learning_rate 1e-4 --model CafMambaBimodal --dataset dvlog --gpu 0 --scheduler
"""

import argparse
import os
import yaml

import wandb
import torch
from tqdm import tqdm

from models.CafMamba import CafMamba
from models.CafMambaBimodal import CafMambaBimodal
from datasets import get_dvlog_dataloader, get_lmvd_dataloader


CONFIG_PATH = "./config/config.yaml"
NAME = "name_dir"


def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Train and test a model.")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
    )
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-lr_decay", "--scheduler", action="store_true")
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("-g", "--gpu", type=str)
    parser.add_argument("-wdb", "--if_wandb", action="store_true")
    parser.add_argument("-tqdm", "--tqdm_able", type=bool)
    parser.add_argument("-tr", "--train", action="store_true")
    parser.add_argument("-d", "--device", type=str, nargs="*")
    parser.set_defaults(**config)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args


def train_epoch(
    net,
    train_loader,
    loss_fn,
    optimizer,
    device,
    current_epoch,
    total_epochs,
    tqdm_able,
):
    """
    Train the model for one epoch.
    """
    net.train()
    sample_count = 0
    running_loss = 0.0
    correct_count = 0

    with tqdm(
        train_loader,
        desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False,
        unit="batch",
        disable=tqdm_able,
    ) as pbar:
        for x, y, mask in pbar:
            x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)

            optimizer.zero_grad()
            y_pred = net(x, mask)
            loss = loss_fn(y_pred, y.to(torch.float32))
            loss.backward()
            optimizer.step()

            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]
            # binary classification
            pred = (y_pred > 0.0).int()
            correct_count += (pred == y).sum().item()

            pbar.set_postfix(
                {
                    "loss": running_loss / sample_count,
                    "acc": correct_count / sample_count,
                }
            )

    return {
        "loss": running_loss / sample_count,
        "acc": correct_count / sample_count,
    }


def val(net, val_loader, loss_fn, device, tqdm_able):
    """
    Test and validate the model.
    """
    net.eval()
    sample_count = 0
    running_loss = 0.0
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able
        ) as pbar:
            for x, y, mask in pbar:
                x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
                y_pred = net(x, mask)
                loss = loss_fn(y_pred, y.to(torch.float32))

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                # binary classification
                pred = (y_pred > 0.0).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                accuracy = (TP + TN) / sample_count if sample_count > 0 else 0.0

                pbar.set_postfix(
                    {
                        "loss": l,
                        "acc": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1_score,
                    }
                )

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (TP + TN) / sample_count if sample_count > 0 else 0.0
    return {
        "loss": l,
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    }


def main():
    args = parse_args()
    args.data_dir = os.path.join(args.data_dir, args.dataset)

    runname = (
        f"{args.dataset}_ep{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}"
    )
    if args.if_wandb:
        wandb_run_name = f"{runname}"
        wandb.init(
            project=NAME,
            entity="your_wandb_entity",
            config=args,
            name=wandb_run_name,
        )

    # build the save directory
    if args.train:
        os.makedirs(f"{args.save_dir}/{args.dataset}_{NAME}", exist_ok=True)
        os.makedirs(f"{args.save_dir}/{args.dataset}_{NAME}/{runname}", exist_ok=True)

    # choose the model for multimodal or bimodal features
    if args.dataset == "lmvd" and args.model == "CafMamba":
        net = CafMamba(**args.cafmamba_lmvd)
    elif args.dataset == "lmvd" and args.model == "CafMambaBimodal":
        net = CafMambaBimodal(**args.cafmambabi_lmvd)
    elif args.dataset == "dvlog":
        net = CafMambaBimodal(**args.cafmambabi_dvlog)

    net = net.to(args.device[0])
    if len(args.device) > 1:
        net = torch.nn.DataParallel(net, device_ids=args.device)

    # get dataloader
    if args.dataset == "dvlog":
        train_loader = get_dvlog_dataloader(args.data_dir, "train", args.batch_size)
        val_loader = get_dvlog_dataloader(args.data_dir, "valid", args.batch_size)
        test_loader = get_dvlog_dataloader(args.data_dir, "test", args.batch_size)
    elif args.dataset == "lmvd":
        train_loader = get_lmvd_dataloader(args.data_dir, "train", args.batch_size)
        val_loader = get_lmvd_dataloader(args.data_dir, "valid", args.batch_size)
        test_loader = get_lmvd_dataloader(args.data_dir, "test", args.batch_size)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    if args.scheduler:
        if args.dataset == "lmvd":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.6, patience=5, verbose=True
            )
        if args.dataset == "dvlog":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.95, patience=5, verbose=True
            )

    best_val_acc = -1.0
    best_acc_epoch = 0

    if args.train:
        for epoch in range(args.epochs):
            train_results = train_epoch(
                net,
                train_loader,
                loss_fn,
                optimizer,
                args.device[0],
                epoch,
                args.epochs,
                args.tqdm_able,
            )
            val_results = val(net, val_loader, loss_fn, args.device[0], args.tqdm_able)
            test_results = val(
                net, test_loader, loss_fn, args.device[0], args.tqdm_able
            )

            val_acc = (
                val_results["acc"]
                + val_results["precision"]
                + val_results["recall"]
                + val_results["f1"]
            ) / 4.0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_acc_epoch = epoch
                torch.save(
                    net.state_dict(),
                    f"{args.save_dir}/{args.dataset}_{NAME}/{runname}/{str(best_acc_epoch)}_best.pt",
                )

            if args.if_wandb:
                wandb.log(
                    {
                        "loss/train": train_results["loss"],
                        "acc/train": train_results["acc"],
                        "loss/val": val_results["loss"],
                        "acc/val": val_results["acc"],
                        "precision/val": val_results["precision"],
                        "recall/val": val_results["recall"],
                        "f1/val": val_results["f1"],
                    }
                )
            if args.scheduler:
                scheduler.step(val_results["loss"])

    # load the best model and test using test set
    with torch.no_grad():
        net.load_state_dict(
            torch.load(
                f"{args.save_dir}/{args.dataset}_{NAME}/{runname}/{str(best_acc_epoch)}_best.pt",
                map_location=args.device[0],
            )
        )
        net.eval()
        test_results = val(net, test_loader, loss_fn, args.device[0], args.tqdm_able)
        print("Test results:")
        print(test_results)

        with open(f"./results/{args.dataset}_{NAME}_{runname}_best_val.txt", "w") as f:
            test_result_str = f'Accuracy:{test_results["acc"]}, Precision:{test_results["precision"]}, Recall:{test_results["recall"]}, F1:{test_results["f1"]}, Avg:{(test_results["acc"] + test_results["precision"]+ test_results["recall"]+ test_results["f1"])/4.0}'
            f.write(test_result_str)

    # upload the best model and test results to wandb
    if args.if_wandb and args.train:
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(
            f"{args.save_dir}/{args.dataset}_{NAME}/{runname}/{str(best_acc_epoch)}_best.pt"
        )
        wandb.run.summary["acc/best_val_acc"] = best_val_acc
        wandb.log_artifact(artifact)
        wandb.run.summary["acc/test_acc"] = test_results["acc"]
        wandb.run.summary["loss/test_loss"] = test_results["loss"]
        wandb.run.summary["precision/test_precision"] = test_results["precision"]
        wandb.run.summary["recall/test_recall"] = test_results["recall"]
        wandb.run.summary["f1/test_f1"] = test_results["f1"]

        wandb.finish()


if __name__ == "__main__":
    main()
