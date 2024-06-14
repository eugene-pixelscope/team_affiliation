from argparse import ArgumentTypeError

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm
import cv2
from PIL import Image
from glob import glob
from torchvision import tv_tensors
from pycocotools.coco import COCO
from pathlib import Path

class HockeyDataset(Dataset):
    def __init__(self, anno_path, transform=None):
        self.transform = transform
        self.anno_path = anno_path
        self.root_img_path = os.path.join(Path(os.path.dirname(self.anno_path)).parent, 'images')

        self.game_names = []
        self.use_ids = []
        self.use_data = []
        self.n_games = 1
        self.n_cat = self.n_games * 2

        self.data = self.load_data(anno_path)

    def load_data(self, anno_path):
        data = []
        # with open(anno_path, 'r') as f:
        #     json_data = json.load(f)
        db = COCO(anno_path)
        failed_img_cnt = 0
        try:
            for k, ann in tqdm(db.anns.items()):
                img_id = ann['image_id']
                x,y,w,h = ann['bbox']
                x1,y1,x2,y2 = x,y,x+w,y+h
                super_cat_id = ann['category_id']
                cat_id = 0
                img_path = os.path.join(self.root_img_path, db.loadImgs(img_id)[0]['file_name'])
                self.load_image(img_path)

                data.append({
                    'img_path': img_path,
                    'img_id': img_id,
                    'super_cat_id': super_cat_id,
                    'cat_id': cat_id,
                    'bbox': [x1,y1,x2,y2],
                })
        except FileNotFoundError as e:
            print(e)
            failed_img_cnt += 1
        print('Data loading end.\n'
              f'The number of images: {len(data)} \n'
              f'The number of loading failed images: {failed_img_cnt}')
        return data

    @staticmethod
    def load_image(path, type='PIL'):
        if type == 'PIL':
            _img = Image.open(path).convert('RGB')
        elif type == 'opencv':
            _img = cv2.imread(path, cv2.IMREAD_COLOR)
            if _img is None:
                raise FileNotFoundError(f'{path} is not a valid image')
        else:
            raise ArgumentTypeError('Type must be PIL or opencv')
        return _img

    @staticmethod
    def load_mask(path):
        _img = Image.open(path)
        return _img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anno = self.data[idx].copy()
        bbox = anno['bbox']
        img = self.load_image(anno['img_path'])
        croped = img.crop(bbox)
        cat = anno['cat_id']

        if self.transform is None:
            return croped, cat
        else:
            input_tensor = self.transform(croped)
            return input_tensor, cat

if __name__ == '__main__':
    from modules import transform
    dataset = HockeyDataset(
        anno_path='datasets/hockey/BRAX001/annotations/instances_BRAX001.json',
        transform=transform.Transforms(size=224),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        # drop_last=True,
        num_workers=2,
    )
    for step, (img, gc) in enumerate(data_loader):
        # B, C, H, W
        B, _, _, _ = img.size()
        for b in range(B):
            vis_img = img[b].permute(1, 2, 0).detach().cpu().numpy() * 255
            cv2.imwrite(f'output/{step}_{b}_{int([gc[b]][0])}.png', cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        if step == 2:
            break


