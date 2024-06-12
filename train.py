import os
from tqdm import tqdm
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
from torchvision.datasets import ImageFolder

from modules import transform, resnet
from modules.loss import TripletLoss, AttentionLoss
from modules.models import PRTreIDTeamClassifier

from data.soccernet_dataset import SoccerNetDataset
from data.ice_hockey_dataset import IceHockeyDataset
from data.soccernetgs_dataset import SoccerNetGSDataset

from utils import yaml_config_hook
from utils.model_utils import save_model, load_model


def train(model, device, data_loader, loss_funcs, wandb=None):
    loss_epoch = 0
    for step, (img, gc) in enumerate(data_loader):
        optimizer.zero_grad()
        # model forward
        img = img.to(device)
        gc = gc.to(device)
        embedding, team_cls_score, _, _ = model(img)

        triplet_loss_fn, identity_loss_fn = loss_funcs
        loss_triplet = triplet_loss_fn(embedding, gc)
        loss_identity = identity_loss_fn(team_cls_score, gc)

        loss = loss_triplet + loss_identity
        loss.backward()
        optimizer.step()

        # logging
        if wandb is not None:
            wandb.log({
                'loss_triplet': loss_triplet.item(),
                'loss_identity': loss_identity.item(),
                'loss': loss.item()
            })
        loss_epoch += loss.item()
    return loss_epoch


def train_with_attention(model, device, data_loader, loss_funcs, attention_weights=0.7, wandb=None):
    loss_epoch = 0

    for step, (img, msk, gc) in enumerate(data_loader):
        optimizer.zero_grad()

        img = img.to(device)
        gc = gc.to(device)
        msk = msk.to(device)
        # model forward
        embedding, team_cls_score, attention, _ = model(img)
        triplet_loss_fn, identity_loss_fn, attention_loss_fn = loss_funcs
        loss_triplet = triplet_loss_fn(embedding, gc)
        loss_identity = identity_loss_fn(team_cls_score, gc.long())

        # resize external masks to fit feature map size
        gt_attention_masks = nn.functional.interpolate(
            msk,
            attention.shape[1::],
            mode="bilinear",
            align_corners=True,
        )
        gt_attention_masks = gt_attention_masks.squeeze(1)
        loss_attention = attention_loss_fn(attention, gt_attention_masks)

        loss = loss_triplet + loss_identity + loss_attention * attention_weights
        loss.backward()
        optimizer.step()

        # logging
        if wandb is not None:
            wandb.log({
                'loss_triplet': loss_triplet.item(),
                'loss_attention': loss_attention.item(),
                'loss_identity': loss_identity.item(),
                'loss': loss.item()
            })
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--use_crop_data', action='store_true')
    args = parser.parse_args()
    config = yaml_config_hook(args.config_file)
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.image_size = eval(str(args.image_size))

    # **********************
    # 0. Initialize setting
    # **********************
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_gpus = torch.cuda.device_count()
    device_ids = [int(x) for x in args.gpus.split(',')]
    print(f'device_ids: {device_ids}')
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    if args.use_wandb:
        import wandb

        wandb.init(project='Team-Affiliation', name=f'{args.model_type}:{args.backbone}:{args.dataset}')
        wandb.config.update({
            'batch_size': args.batch_size,
            'image_size': args.image_size,
            'epochs': args.epochs,
            'workers': args.workers,
            'dataset': args.dataset,
            'backbone': args.backbone,
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay,
            'attention_enable': args.attention_enable,
            'attention_learnable': args.attention_learnable
        })
    else:
        wandb = None

    # **********************
    # 1. Prepare datasets
    # **********************
    if args.dataset == "SoccerNet":
        dataset = SoccerNetDataset(
            anno_path=f'{args.dataset_dir}/soccernet/reid/valid/bbox_info.json',
            transform=transform.TrainTransforms(size=args.image_size),
            subset='valid',
            subtype='gallery'
        )
        n_cat = dataset.n_cat
    elif args.dataset == "IceHockey":
        dataset = IceHockeyDataset(
            root_dir=f'{args.dataset_dir}/ice_hockey',
            transform=transform.TrainTransforms(size=args.image_size),
        )
        n_cat = dataset.n_cat
    elif args.dataset == "SoccerNetGS":
        if not args.use_crop_data:
            dataset = SoccerNetGSDataset(
                root_dir='datasets/SoccerNetGS/gamestate-2024',
                transform=transform.TrainTransforms(size=args.image_size),
                subset='total'
            )
        else:
            dataset = ImageFolder(
                root="/workspace/Contrastive-Clustering/datasets/SoccerNetGS/gamestate-2024/crop",
                transform=transform.TrainTransforms(size=(256, 128)))
        n_cat = 12
    else:
        raise NotImplementedError
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    # **********************
    # 2. Model construct
    # **********************
    if 'res' in str(args.backbone).lower():
        backbone = resnet.get_resnet(args.backbone, pretrained=False)
    else:
        raise ValueError(f'{args.backbone} is not supported yet.')

    model = PRTreIDTeamClassifier(backbone=backbone,
                                  num_teams=n_cat,
                                  num_role=2,
                                  attention_enable=args.attention_enable)

    # Make multi-gpu setting
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    # **********************
    # 3. Loss function and Optimizer setting
    # **********************
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.checkpoint:
        model_fp = os.path.join(args.checkpoint)
        checkpoint = torch.load(model_fp)
        # model load
        if args.changed_datasets:
            load_model(net=model, checkpoint=checkpoint, filter_team_classifier=True)
        else:
            load_model(net=model, checkpoint=checkpoint)
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.attention_learnable:
        attention_loss = AttentionLoss()
    triplet_loss = TripletLoss()
    identity_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    # **********************
    # 4. Model training
    # **********************
    total_epoch = args.start_epoch
    print('Start training ...')
    if args.attention_learnable:
        # start training
        for epoch in tqdm(range(args.start_epoch, args.epochs)):
            lr = optimizer.param_groups[0]["lr"]
            loss_epoch = train_with_attention(
                model,
                device,
                data_loader,
                loss_funcs=(triplet_loss, identity_loss, attention_loss),
                wandb=wandb,
                attention_weights=args.attention_weights)
            # model save
            if epoch % 20 == 0:
                save_model(args, model, optimizer, total_epoch)
            total_epoch += 1
    else:
        # start training
        for epoch in tqdm(range(args.start_epoch, args.epochs)):
            lr = optimizer.param_groups[0]["lr"]
            loss_epoch = train(
                model,
                device,
                data_loader,
                loss_funcs=(triplet_loss, identity_loss),
                wandb=wandb)

            # model save
            if epoch % 20 == 0:
                save_model(args, model, optimizer, total_epoch)
            total_epoch += 1
    save_model(args, model, optimizer, total_epoch)
    print('... End process')
