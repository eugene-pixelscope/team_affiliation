import os
import argparse
import numpy as np
import copy

import torch
from torch.utils.data import DataLoader
import torchvision

from utils import yaml_config_hook
from utils.model_utils import load_model

from modules import resnet, transform
from modules.models import PRTreIDTeamClassifier
from evaluation import evaluation

from data.soccernet_dataset import SoccerNetDataset
from data.ice_hockey_dataset import IceHockeyDataset
from data.soccernetgs_dataset import SoccerNetGSDataset
from data.hockey_dataset import HockeyDataset

from fast_pytorch_kmeans import KMeans


def inference(loader, model, device, n_cluster=2):
    model.eval()
    labels = []
    pred = []
    for step, (x, _, y) in enumerate(loader):
    # for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            f, _, _ = model(x)
        # kmeans
        kmeans = KMeans(n_clusters=n_cluster, mode='euclidean', verbose=0, max_iter=20)
        c = kmeans.fit_predict(f)
        pred.extend(c.detach().cpu().numpy())
        labels.extend(y.numpy())
    return np.array(pred), np.array(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['Hockey', 'SoccerNetGS', 'IceHockey'])
    args = parser.parse_args()
    config = yaml_config_hook(args.config_file)
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()
    args.image_size = eval(str(args.image_size))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "SoccerNet":
        dataset = SoccerNetDataset(
            anno_path=f'{args.dataset_dir}/soccernet/reid/valid/bbox_info.json',
            transform=transform.TransformsForInfer(size=args.image_size),
            subset='valid',
            subtype='gallery'
        )
        n_cat = dataset.n_cat
    elif args.dataset == "IceHockey":
        dataset = IceHockeyDataset(
            root_dir=f'{args.dataset_dir}/ice_hockey',
            transform=transform.TransformsForInfer(size=args.image_size),
        )
        n_cat = dataset.n_cat
    elif args.dataset == "SoccerNetGS":
        dataset = SoccerNetGSDataset(
            root_dir='datasets/SoccerNetGS/gamestate-2024',
            transform=transform.TransformsForInfer(size=args.image_size),
            subset='total'
        )
        n_cat = 12
    elif args.dataset == 'Hockey':
        dataset = HockeyDataset(
            anno_path='datasets/hockey/BRAX001/annotations/instances_BRAX001.json',
            transform=transform.TransformsForInfer(size=args.image_size),
        )
    else:
        raise NotImplementedError
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True
    )

    if 'res' in str(args.backbone).lower():
        backbone = resnet.get_resnet(args.backbone, pretrained=False)
    else:
        raise ValueError(f'{args.backbone} is not supported yet.')

    model = PRTreIDTeamClassifier(backbone=backbone,
                                  num_role=2,
                                  attention_enable=args.attention_enable)

    model_fp = os.path.join(args.checkpoint)
    checkpoint = torch.load(model_fp)
    load_model(net=model, checkpoint=checkpoint, filter_team_classifier=True)
    model.to(device)

    print("### Creating features from model ###")
    X, Y = inference(data_loader, model, device, n_cluster=dataset.n_cat)
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    print('Model Inference [Triplet]')
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
