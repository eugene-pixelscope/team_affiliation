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

from data.hockey_dataset import HockeyDataset_ForRoleCls

from utils import yaml_config_hook
from utils.model_utils import save_model, load_model


def train(model, device, data_loader, optimizer, loss_funcs, wandb=None):
    loss_epoch = 0
    # Model freeze except role-classifier
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'role_classifier' in name:
            continue
        param.requires_grad = False

    for step, (img, gc) in enumerate(data_loader):
        optimizer.zero_grad()
        # model forward
        img = img.to(device)
        gc = gc.to(device)
        _,_,_, role_cls = model(img)

        loss = loss_funcs(role_cls, gc)
        loss.backward()
        optimizer.step()

        # logging
        if wandb is not None:
            wandb.log({
                'loss': loss.item()
            })
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--config_file', type=str, required=True)
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
    if args.dataset == "Hockey":
        dataset = HockeyDataset_ForRoleCls(
            anno_path='datasets/hockey/BRAX001/annotations/instances_BRAX001.json',
            transform=transform.TargetTransforms(size=args.image_size),
        )
        n_cat = dataset.n_cat
    else:
        raise NotImplementedError
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
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

    # Make multi-gpu setting
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    role_cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    # **********************
    # 4. Model training
    # **********************
    total_epoch = args.start_epoch
    print('Start training ...')
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(
            model,
            device,
            data_loader,
            optimizer,
            loss_funcs=role_cls_loss,
            wandb=wandb)

        # model save
        if epoch % 20 == 0:
            save_model(args, model, optimizer, f'{total_epoch}_with_role')
        total_epoch += 1
    save_model(args, model, optimizer, f'{total_epoch}_with_role')
    print('... End process')
