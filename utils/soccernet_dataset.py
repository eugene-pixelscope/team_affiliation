from argparse import ArgumentTypeError

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import cv2
import os.path as osp
import os
from tqdm import tqdm
from PIL import Image

super_cat = [
    'Player_team_left',
    'Player_team_right',
    # 'Goalkeeper_team_left',
    # 'Goalkeeper_team_right',
    # 'Main_referee',
    # 'Side_referee'
]
n_cat = 2


# class SoccerNetDataset(Dataset):
#     def __init__(self, anno_path, subset='train', transform=None, subtype=None):
#         assert subset in ['train', 'valid']
#         self.subset = subset
#         self.transform = transform
#         self.subtype = subtype
#         self.img_root = f'datasets/soccernet/reid/{subset}' if self.subset == 'train' else f'datasets/soccernet/reid/{subset}/{self.subtype}'
#         self.game_names = []
#         self.use_data = []
#         self._std_game_id = 0
#
#         self.data = self.load_data(anno_path)
#         self.prepare()
#
#     @property
#     def std_game_id(self):
#         return self._std_game_id
#
#     @std_game_id.setter
#     def std_game_id(self, game_id):
#         if game_id > len(self.game_names):
#             raise ValueError("game_id out of range")
#         self._std_game_id = game_id
#
#     def load_data(self, anno_path):
#         with open(anno_path, encoding='latin1') as f:
#             db = json.load(f)
#         if self.subset == 'valid':
#             db = db[self.subtype]
#         data = []
#         failed_img_cnt = 0
#         img_id = 0
#         for idx in tqdm(db.keys()):
#             try:
#                 anno = db[idx]
#                 if anno['clazz'] not in super_cat:
#                     continue
#                 dir_path = osp.join(self.img_root, anno['relative_path'])
#                 filename = (f'{anno["bbox_idx"]}-{anno["action_idx"]}-{anno["person_uid"]}-{anno["frame_idx"]}'
#                             f'-{anno["clazz"]}-{anno["id"]}-{anno["UAI"]}-{anno["height"]}x{anno["width"]}.png')
#                 img_path = osp.join(dir_path, filename)
#                 self.load_image(img_path)
#                 game_date = ''.join(anno['relative_path'].split('/')[-2].split(' ')[:3])
#                 if game_date not in self.game_names:
#                     self.game_names.append(game_date)
#                 # game_id = self.game_names.index(game_date)
#                 # concat 'Main_referee' with 'Side_referee'
#                 super_cat_id = super_cat.index(anno['clazz']) % n_cat
#
#                 data.append({
#                     'img_path': img_path,
#                     'img_id': img_id,
#                     # 'game_id': game_id,
#                     'game_name': game_date,
#                     'super_cat_id': super_cat_id
#                 })
#                 img_id += 1
#             except FileNotFoundError as e:
#                 print(e)
#                 failed_img_cnt += 1
#                 continue
#         self.game_names = sorted(self.game_names)
#         for d in data:
#             d.update({
#                 'game_id': self.game_names.index(d['game_name']),
#             })
#         print('Data loading end.\n'
#               f'The number of images: {len(data)} \n'
#               f'The number of loading failed images: {failed_img_cnt}')
#         return data
#
#     def get_id_by_game_id(self, game_id):
#         return [x for x in self.data if x['game_id'] == game_id]
#
#     def prepare(self):
#         self.use_data = self.get_id_by_game_id(self.std_game_id)
#
#     @staticmethod
#     def load_image(path, type='PIL'):
#         if type == 'PIL':
#             _img = Image.open(path).convert('RGB')
#         elif type == 'opencv':
#             _img = cv2.imread(path, cv2.IMREAD_COLOR)
#             if _img is None:
#                 raise FileNotFoundError(f'{path} is not a valid image')
#         else:
#             raise ArgumentTypeError('Type must be PIL or opencv')
#         return _img
#
#     def __len__(self):
#         return len(self.use_data)
#
#     def __getitem__(self, idx):
#         anno = self.use_data[idx].copy()
#         img = self.load_image(anno['img_path'])
#         cat = anno['super_cat_id']
#         if self.transform is None:
#             return img, cat
#         else:
#             input_tensor = self.transform(img)
#             return input_tensor, cat


class SoccerNetDataset(Dataset):
    def __init__(self, anno_path, subset='train', transform=None, subtype=None):
        assert subset in ['train', 'valid']
        self.subset = subset
        self.transform = transform
        self.subtype = subtype
        self.img_root = f'datasets/soccernet/reid/{subset}' if self.subset == 'train' else f'datasets/soccernet/reid/{subset}/{self.subtype}'

        self.game_names = []
        self.n_games = 2
        self.n_cat = 3 * 2

        self.data = self.load_data(anno_path)

    def load_data(self, anno_path):
        with open(anno_path, encoding='latin1') as f:
            db = json.load(f)
        if self.subset == 'valid':
            db = db[self.subtype]

        data = []
        failed_img_cnt = 0
        img_id = 0
        for idx in tqdm(db.keys()):
            try:
                anno = db[idx]
                super_cat_nm = anno['clazz']
                if super_cat_nm not in super_cat:
                    continue
                dir_path = osp.join(self.img_root, anno['relative_path'])
                g_name = dir_path.split('/')[-2]
                a_team = ' '.join(g_name.split(' - ')[1].split(' ')[1:-1])
                b_team = ' '.join(g_name.split(' - ')[-1].split(' ')[1:])
                filename = (f'{anno["bbox_idx"]}-{anno["action_idx"]}-{anno["person_uid"]}-{anno["frame_idx"]}'
                            f'-{anno["clazz"]}-{anno["id"]}-{anno["UAI"]}-{anno["height"]}x{anno["width"]}.png')
                img_path = osp.join(dir_path, filename)
                self.load_image(img_path)
                game_date = ''.join(anno['relative_path'].split('/')[-2].split(' ')[:3])
                if game_date not in self.game_names:
                    # TODO:
                    # if len(self.game_names) > self.n_games:
                    #     continue
                    self.game_names.append(game_date)
                # game_id = self.game_names.index(game_date)
                # concat 'Main_referee' with 'Side_referee'
                super_cat_id = super_cat.index(super_cat_nm) % n_cat

                data.append({
                    'img_path': img_path,
                    'img_id': img_id,
                    'game_name': game_date,
                    'super_cat_id': super_cat_id,
                    'team_nms': [a_team, b_team]
                })
                img_id += 1
            except FileNotFoundError as e:
                print(e)
                failed_img_cnt += 1
                continue
        self.game_names = sorted(self.game_names)
        for d in data:
            game_id = self.game_names.index(d['game_name'])
            d.update({
                'game_id': self.game_names.index(d['game_name']),
                'cat_id': d['super_cat_id'] + game_id * 2,
            })
        print('Data loading end.\n'
              f'The number of images: {len(data)} \n'
              f'The number of loading failed images: {failed_img_cnt}')
        return data

    def get_id_by_game_id(self, game_id):
        return [x for x in self.data if x['game_id'] == game_id]

    def prepare(self):
        self.use_data = self.get_id_by_game_id(self.std_game_id)

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anno = self.data[idx].copy()
        img = self.load_image(anno['img_path'])
        cat = anno['super_cat_id']
        if self.transform is None:
            return img, cat
        else:
            input_tensor = self.transform(img)
            return input_tensor, cat


if __name__ == '__main__':
    from modules import transform
    from torch.utils import data
    dataset = SoccerNetDataset(
        anno_path="datasets/soccernet/reid/valid/bbox_info.json",
        transform=transform.Transforms(size=(256, 128)),
        subtype='gallery',
        subset='valid')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        # shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    for step, (img, gc) in enumerate(data_loader):
        # B, C, H, W
        B, _, _, _ = img.size()
        for b in range(B):
            vis_img = img[b].permute(1, 2, 0).detach().cpu().numpy() * 255
            cv2.imwrite(f'output/{step}_{b}_{super_cat[gc[b]]}.png', cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        if step == 2:
            break