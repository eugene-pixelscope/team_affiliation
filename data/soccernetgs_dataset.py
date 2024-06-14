import copy
from argparse import ArgumentTypeError
from collections import OrderedDict

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
from utils.dir_utils import make_folder

# filter_std = OrderedDict({
#     "39-39": ['right', 'left'],
#     "60-60": ['right', 'left'],
#     "78-78": ['right', 'left'],
#     "97-97": ['right', 'left'],
#     "116-116": ['left'],
#     "132-132": ['right', 'left'],
#     "187-187": ['left'],
# })
filter_std = OrderedDict({
    "39-48": ['right', 'left'],
    "60-68": ['right', 'left'],
    "78-85": ['right', 'left'],
    "97-106": ['right', 'left'],
    "116-123": ['left'],
    "132-140": ['right', 'left'],
    "187-194": ['left'],
})
# filter_std = OrderedDict({
#     "39-59": ['right', 'left'],
#     "60-77": ['right', 'left'],
#     "78-96": ['right', 'left'],
#     "97-115": ['right', 'left'],
#     "116-131": ['left'],
#     "132-150": ['right', 'left'],
#     "187-200": ['left'],
# })

def get_attribute_by_id(game_no):
    for idx, (k,v) in enumerate(filter_std.items()):
        _min, _max = [int(x) for x in k.split('-')]
        if _min <= game_no <= _max:
            return idx, v
    raise ValueError(f'Game number {game_no} not in valid range')

def get_catid_by_index_and_team(idx, team):
    count = 0
    for _idx, (k,v) in enumerate(filter_std.items()):
        for t in v:
            if _idx == idx and t == team:
                return count
            count += 1

class SoccerNetGSDataset(Dataset):
    def __init__(self, root_dir, subset='total', transform=None):
        self.transform = transform
        self.root_dir = os.path.join(root_dir, subset)

        self.game_names = []
        self.use_ids = []
        self.use_data = []
        self.n_cat = 12
        self.data = self.load_data()

    def load_data(self):
        data = []
        db = COCO()
        for idx, dir_path in enumerate([x for x in glob(f'{self.root_dir}/*') if os.path.isdir(x)]):
            anno_path = f'{dir_path}/Labels-GameState.json'
            with open(anno_path, 'r') as f:
                json_data = json.load(f)
            # preprocessing
            for x in json_data['images']:
                x['id'] = int(x['image_id'])
                x.pop('image_id')
                x['im_dir'] = json_data['info']['im_dir']
                x['nm_dir'] = json_data['info']['name']
            for x in json_data['annotations']:
                x['id'] = int(x['id'])
                x['image_id'] = int(x['image_id'])
            json_data['categories'] = [json_data['categories'][0]]

            if len(db.dataset) == 0:
                db.dataset['images'] = json_data['images']
                db.dataset['annotations'] = json_data['annotations']
                db.dataset['categories'] = json_data['categories']
            else:
                db.dataset['images'] += json_data['images']
                db.dataset['annotations'] += json_data['annotations']
        db.createIndex()

        failed_img_cnt = 0
        for k, _ann in tqdm(db.anns.items()):
            try:
                ann = copy.deepcopy(_ann)
                super_cat_id = ann['category_id']
                if super_cat_id!=1:
                    continue

                img_id = ann['image_id']
                img_info = db.loadImgs(img_id)[0]
                game_id = int(img_info['nm_dir'].split('-')[-1])
                idx, std_label = get_attribute_by_id(game_id)
                if ann['attributes']['team'] not in std_label:
                    continue
                bboxes = ann['bbox_image']
                x,y,w,h = bboxes['x'], bboxes['y'], bboxes['w'], bboxes['h']
                x1,y1,x2,y2 = x,y,x+w,y+h
                img_path = os.path.join(self.root_dir,
                                        img_info['nm_dir'],
                                        img_info['im_dir'],
                                        img_info['file_name'])

                data.append({
                    'img_path': img_path,
                    'img_id': img_id,
                    'game_name': img_info['nm_dir'],
                    'super_cat_id': super_cat_id,
                    'cat_id': get_catid_by_index_and_team(idx, ann['attributes']['team']),
                    'game_id': game_id,
                    'bbox': [x1,y1,x2,y2],
                })
            except FileNotFoundError as e:
                print(e)
                failed_img_cnt += 1
            except ValueError as e:
                continue
        self.n_cat = len(set([x['cat_id'] for x in data]))
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
        anno = self.data[idx]
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
    from torchvision.datasets import ImageFolder
    # dataset = SoccerNetGSDataset(
    #     root_dir='/workspace/Contrastive-Clustering/datasets/SoccerNetGS/gamestate-2024',
    #     transform=transform.Transforms(size=(256, 128))
    # )
    dataset = ImageFolder(root="/workspace/Contrastive-Clustering/datasets/SoccerNetGS/gamestate-2024/crop",
                          transform=transform.Transforms(size=(256, 128)))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        # drop_last=True,
        num_workers=2,
    )
    for step, (img, gc) in enumerate(tqdm(data_loader)):
        # B, C, H, W
        B, _, _, _ = img.size()
        for b in range(B):
            vis_img = img[b].permute(1, 2, 0).detach().cpu().numpy() * 255
            dir_path = f'output'
            make_folder(dir_path)
            cv2.imwrite(f'{dir_path}/{b}_{int([gc[b]][0])}.jpg', cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        break

    # count = 0
    # for step, (img, gc) in enumerate(tqdm(data_loader)):
    #     # B, C, H, W
    #     B, _, _, _ = img.size()
    #     for b in range(B):
    #         vis_img = img[b].permute(1, 2, 0).detach().cpu().numpy() * 255
    #         dir_path = f'datasets/SoccerNetGS/gamestate-2024/crop/{int([gc[b]][0])}'
    #         make_folder(dir_path)
    #         cv2.imwrite(f'{dir_path}/{str(count).zfill(8)}.jpg', cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    #         count += 1


