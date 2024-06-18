import copy
import os
import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np

import matplotlib.pyplot as plt
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import seaborn as sns

from modules import resnet, transform, shufflenetv2, efficientnet
from torch.utils.data import DataLoader

from utils import yaml_config_hook
from utils.model_utils import load_model, torch_time_checker
from utils.dir_utils import make_folder

from modules.models import PRTreIDTeamClassifier

from data.soccernet_dataset import SoccerNetDataset
from data.ice_hockey_dataset import IceHockeyDataset
from data.soccernetgs_dataset import SoccerNetGSDataset
from data.hockey_dataset import HockeyDataset

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, FeatureAgglomeration, Birch, OPTICS, SpectralBiclustering, AffinityPropagation
import cv2


@torch_time_checker
def inference(batch_tensor, model, clustering_engine):
    with torch.no_grad():
        f, _, _, _ = model(batch_tensor)
    # clustering
    features = f.detach().cpu().numpy()
    clustering_engine.fit(features)
    return clustering_engine.labels_, f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['Hockey', 'SoccerNetGS'])
    args = parser.parse_args()
    config = yaml_config_hook(args.config_file)
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()
    args.image_size = eval(str(args.image_size))

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

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
        drop_last=True,
    )

    # init tensor (std images)
    totensor = torchvision.transforms.ToTensor()
    init_tensor_cls1 = totensor(np.array(HockeyDataset.load_image('std_imgs/0_1_0.png'))).unsqueeze(0)
    init_tensor_cls2 = totensor(np.array(HockeyDataset.load_image('std_imgs/0_2_0.png'))).unsqueeze(0)
    global init_cluster
    init_cluster = torch.cat((init_tensor_cls1, init_tensor_cls2), dim=0).to(device)

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

    clustering_engine = KMeans(n_clusters=2, random_state=seed, max_iter=20, n_init=10)
    # clustering_engine = Birch(n_clusters=2)
    # clustering_engine = SpectralClustering(n_clusters=2,
    #                                        assign_labels='discretize',
    #                                        random_state=seed)

    # Model Inference
    for step, (x, y) in enumerate(data_loader):
        x = x.to(device)
        labels, features = inference(x, model, clustering_engine=clustering_engine)
        print(labels)

        # ***************
        # Visualization
        # ***************
        B, _, _, _ = x.size()  # B,C,H,W
        make_folder(f'output/{step}/0')
        make_folder(f'output/{step}/1')

        out = pairwise_cosine_similarity(features)
        out.fill_diagonal_(1)
        plt.clf()
        ax = sns.heatmap(out.detach().cpu().numpy(), linewidth=0.5)
        plt.savefig(f'output/{step}/cosine.png')

        for b in range(B):
            vis_img = x[b].permute(1, 2, 0).detach().cpu().numpy() * 255
            cv2.imwrite(f'output/{step}/{int(labels[b])}/{b}.png', cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
