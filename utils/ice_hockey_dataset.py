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

# class IceHockeyDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.transform = transform
#         self.root_dir = root_dir
#         self.game_names = []
#         self.use_ids = []
#         self.use_data = []
#         self._std_game_id = 0
#
#         self.data = self.load_data(root_dir)
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
#     def load_data(self, root_dir):
#         data = []
#         failed_img_cnt = 0
#         img_id = 0
#
#         for dir_name in glob(root_dir + '/*'):
#                 vid_name = dir_name.split('/')[-1].split('-')[0]
#                 with open(os.path.join(dir_name, 'gt.txt'), 'r') as f:
#                     lines = f.readlines()
#                 for line in lines:
#                     img_name, cat = line.split(',')
#                     try:
#                         img_path = os.path.join(dir_name, 'crops', img_name)
#                         mask_path = os.path.join(dir_name, 'masks', img_name)
#
#                         self.load_image(img_path)
#                         self.load_mask(mask_path)
#                         if vid_name not in self.game_names:
#                             self.game_names.append(vid_name)
#
#                         cat_id = int(cat)
#                         if cat_id == 2:
#                             continue
#                         data.append({
#                             'img_path': img_path,
#                             'mask_path': mask_path,
#                             'img_id': img_id,
#                             'game_name': vid_name,
#                             'super_cat_id': cat_id % 3,
#                         })
#                         img_id += 1
#                     except FileNotFoundError as e:
#                         print(e)
#                         failed_img_cnt += 1
#                         continue
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
#     @staticmethod
#     def load_mask(path):
#         _img = Image.open(path)
#         return _img
#
#     def __len__(self):
#         return len(self.use_data)
#
#     def __getitem__(self, idx):
#         anno = self.use_data[idx].copy()
#         img = self.load_image(anno['img_path'])
#         msks = self.load_mask(anno['mask_path'])
#         cat = anno['super_cat_id']
#
#         masks = tv_tensors.Mask(msks)
#
#         if self.transform is None:
#             return img, msks, cat
#         else:
#             input_tensor, input_mask = self.transform((img, masks))
#             input_mask = input_mask / 255.
#             return input_tensor, input_mask, cat

class IceHockeyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir

        self.game_names = []
        self.n_games = 2
        self.n_cat = self.n_games * 2
        self.data = self.load_data(root_dir)

    def load_data(self, root_dir):
        data = []
        failed_img_cnt = 0
        img_id = 0

        for dir_idx, dir_name in enumerate(glob(root_dir + '/*')):
            if dir_idx > self.n_games:
                break
            vid_name = dir_name.split('/')[-1].split('-')[0]
            with open(os.path.join(dir_name, 'gt.txt'), 'r') as f:
                lines = f.readlines()
            for line in lines:
                img_name, super_cat_no = line.split(',')
                try:
                    super_cat_id = int(super_cat_no)
                    if super_cat_id == 2:
                        continue

                    img_path = os.path.join(dir_name, 'crops', img_name)
                    mask_path = os.path.join(dir_name, 'masks', img_name)

                    self.load_image(img_path)
                    self.load_mask(mask_path)
                    if vid_name not in self.game_names:
                        self.game_names.append(vid_name)

                    data.append({
                        'img_path': img_path,
                        'mask_path': mask_path,
                        'img_id': img_id,
                        'game_name': vid_name,
                        'super_cat_id': super_cat_id % 3,
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
                'game_id': game_id,
                'cat_id': d['super_cat_id'] + game_id * 2,
            })
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
        img = self.load_image(anno['img_path'])
        msks = self.load_mask(anno['mask_path'])
        cat = anno['cat_id']

        masks = tv_tensors.Mask(msks)

        if self.transform is None:
            return img, msks, cat
        else:
            input_tensor, input_mask = self.transform((img, masks))
            input_mask = input_mask / 255.
            return input_tensor, input_mask, cat

if __name__ == '__main__':
    from modules import transform
    dataset = IceHockeyDataset(
        root_dir='datasets/ice_hockey',
        # transform=transform.Transforms(size=(256, 128), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transform=transform.Transforms(size=(256, 128)),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        # drop_last=True,
        num_workers=2,
    )
    for step, (img, msk, gc) in enumerate(data_loader):
        # B, C, H, W
        B, _, _, _ = img.size()
        for b in range(B):
            vis_img = img[b].permute(1, 2, 0).detach().cpu().numpy() * 255
            vis_msk_img = msk[b].permute(1, 2, 0).detach().cpu().numpy() * 255
            cv2.imwrite(f'output/{step}_{b}_{int([gc[b]][0])}.png', cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            cv2.imwrite(f'output/{step}_{b}_{int([gc[b]][0])}_msk.png', cv2.cvtColor(vis_msk_img, cv2.COLOR_BGR2RGB))
        if step == 2:
            break


