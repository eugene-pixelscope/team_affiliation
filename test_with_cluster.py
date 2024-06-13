import os
import argparse
import torch
import torchvision
import numpy as np

from modules import resnet, transform
from torch.utils.data import DataLoader

from utils import yaml_config_hook
from utils.model_utils import load_model
from utils.dir_utils import make_folder

from modules.models import PRTreIDTeamClassifier

from data.soccernet_dataset import SoccerNetDataset
from data.ice_hockey_dataset import IceHockeyDataset
from data.soccernetgs_dataset import SoccerNetGSDataset
from data.hockey_dataset import HockeyDataset

from fast_pytorch_kmeans import KMeans

import cv2



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    # model.eval()
    # labels = []
    # # init cluster
    # with torch.no_grad():
    #     x = init_cluster.to(device)
    #     init_f, _, _ = model(x)
    # for step, (x, y) in enumerate(loader):
    #     x = x.to(device)
    #     with torch.no_grad():
    #         f, _, _ = model(x)
    #     # kmeans
    #     fs = torch.cat([init_f, f], dim=0)
    #     cluster_ids_x, cluster_centers = kmeans(
    #         X=fs, num_clusters=2, distance='cosine', device=device, tol=1e-3
    #     )
def inference(loader, model, device):
    model.eval()
    pred = []
    with torch.no_grad():
        x = init_cluster.to(device)
        init_f, _, _ = model(x)
    # for step, (x, _, y) in enumerate(loader):
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            f, _, _ = model(x)
        fs = torch.cat([init_f, f], dim=0)

        # kmeans
        kmeans = KMeans(n_clusters=2, mode='euclidean', verbose=0, max_iter=20)
        c = kmeans.fit_predict(fs)

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

        l_team = int(c[0])
        r_team = int(not c[0])
        pred.extend(c[2:].detach().cpu().numpy().tolist())
        # vis
        B, _, _, _ = x.size() # B,C,H,W
        make_folder(f'output/Left')
        make_folder(f'output/Right')
        for b in range(B):
            # vis_img = unnormalize(x[b]).permute(1, 2, 0).detach().cpu().numpy() * 255
            vis_img = x[b].permute(1, 2, 0).detach().cpu().numpy() * 255
            if l_team == int(c[b+2]):
                cv2.imwrite(f'output/Left/{step}_{b+2}.png', cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            elif r_team == int(c[b+2]):
                cv2.imwrite(f'output/Right/{step}_{b+2}.png', cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        break
    return pred


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
    init_cluster = torch.cat((init_tensor_cls1, init_tensor_cls2), dim=0)

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

    labels = inference(data_loader, model, device)
    print(labels)
