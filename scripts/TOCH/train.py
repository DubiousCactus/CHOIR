import argparse
import os

import numpy as np
import torch
from torch import nn, optim
from train_fn import train_model

from dataset.torch_loaders import ContactPoseDataset, GRAB_Dataset
from model.TOCH.model import TemporalPointAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--ckpt_path", default="./ckpt", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_epoch", default=400, type=int)
    parser.add_argument("--lr", default=8e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--num_gpu", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--latent_size", default=64, type=int)
    parser.add_argument("--window_size", default=30, type=int)
    parser.add_argument("--step_size", default=15, type=int)
    parser.add_argument("--num_worker", default=16, type=int)
    parser.add_argument(
        "--dataset", default="contactpose", type=str, choices=["grab", "contactpose"]
    )
    # save all the intermediate ckeckpoints
    parser.add_argument("--save_int_ckpt", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_printoptions(precision=6)

    # config gpu and model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0")
    if args.num_gpu > 1:
        device_ids = list(range(args.num_gpu))
        ho_autoencoder = nn.DataParallel(
            TemporalPointAE(
                input_dim=11, latent_dim=args.latent_size, window_size=args.window_size
            ),
            device_ids=device_ids,
        ).to(device)
    else:
        ho_autoencoder = TemporalPointAE(
            input_dim=11, latent_dim=args.latent_size, window_size=args.window_size
        ).to(device)

    # setup optimizer
    opt = optim.Adam(
        list(ho_autoencoder.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )

    # prepare data
    if args.dataset == "grab":
        train_set = GRAB_Dataset(
            args.data_path,
            "train",
            num_points=8000,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        vald_set = GRAB_Dataset(
            args.data_path,
            "val",
            num_points=8000,
            window_size=args.window_size,
            step_size=args.step_size,
        )
    elif args.dataset == "contactpose":
        train_set = ContactPoseDataset(args.data_path, "train")
        vald_set = ContactPoseDataset(args.data_path, "val")
    else:
        raise NotImplementedError("Dataset not implemented")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker
    )
    vald_loader = torch.utils.data.DataLoader(
        vald_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker
    )

    if not os.path.isdir(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    train_model(ho_autoencoder, train_loader, vald_loader, opt, device, args)
