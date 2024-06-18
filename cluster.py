import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

from utils import yaml_config_hook
from utils.model_utils import load_model, torch_time_checker

from modules import resnet, transform, shufflenetv2, efficientnet
from modules.models import PRTreIDTeamClassifier
from evaluation import evaluation

from data.soccernet_dataset import SoccerNetDataset
from data.ice_hockey_dataset import IceHockeyDataset
from data.soccernetgs_dataset import SoccerNetGSDataset
from data.hockey_dataset import HockeyDataset

from sklearn.cluster import KMeans, SpectralClustering, Birch


@torch_time_checker
def inference(batch_tensor, model):
    with torch.no_grad():
        f, _, _, _ = model(batch_tensor)
    return f.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['Hockey', 'SoccerNetGS', 'IceHockey'])
    parser.add_argument('--use_crop_data', action='store_true')
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
        if not args.use_crop_data:
            dataset = HockeyDataset(
                anno_path='datasets/hockey/BRAX001/annotations/instances_BRAX001.json',
                transform=transform.TransformsForInfer(size=args.image_size),
            )
        else:
            total_dataset = ImageFolder(
                root='datasets/hockey/BRAX001/crop',
                transform=transform.TransformsForInfer(size=args.image_size))
            idx = [i for i in range(len(total_dataset)) if total_dataset.imgs[i][1] != total_dataset.class_to_idx['2']]
            # build the appropriate subset
            dataset = Subset(total_dataset, idx)
            n_cat = 2
    else:
        raise NotImplementedError
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False
    )

    if 'resnet' in str(args.backbone).lower():
        backbone = resnet.get_resnet(args.backbone, pretrained=False)
    elif 'shufflenet' in str(args.backbone).lower():
        backbone = shufflenetv2.get_shufflenet(args.backbone)
    elif 'efficientnet' in str(args.backbone).lower():
        backbone = efficientnet.get_efficientnet(args.backbone)
    else:
        raise ValueError(f'{args.backbone} is not supported yet.')

    model = PRTreIDTeamClassifier(backbone=backbone,
                                  num_role=2,
                                  attention_enable=args.attention_enable)

    model_fp = os.path.join(args.checkpoint)
    checkpoint = torch.load(model_fp)
    load_model(net=model, checkpoint=checkpoint, filter_team_classifier=True)
    model.to(device)
    model.eval()

    clustering_engine = KMeans(n_clusters=n_cat, random_state=args.seed, max_iter=20, n_init=10)
    # clustering_engine = Birch(n_clusters=n_cat)
    # clustering_engine = SpectralClustering(n_clusters=n_cat,
    #                                        assign_labels='discretize',
    #                                        random_state=args.seed)

    print("### Creating features from model ###")
    X = []
    Y = []
    for step, batch in enumerate(data_loader):
        x, y = batch[0], batch[-1]
        x = x.to(device)
        features = inference(x, model)
        X.extend(features)
        Y.extend(y.detach().cpu().numpy())

    clustering_engine.fit(X)
    pred_cluster = clustering_engine.labels_
    nmi, ari, f, acc = evaluation.evaluate(Y, pred_cluster)
    print('Model Inference [Triplet]')
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
