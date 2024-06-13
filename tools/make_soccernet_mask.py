import os
from argparse import ArgumentTypeError
from glob import glob
import shutil
from pycocotools.coco import COCO
from PIL import Image
import cv2
import json

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

ROOT_PATH = '/workspace/data/SoccerNetGS/gamestate-2024/total'
OUTPUT_PATH = '/workspace/data/output/croped'
for dir_path in glob(f'{ROOT_PATH}/*'):
    dir_name = dir_path.split('/')[-1]
    anno_path = f'{dir_path}/Labels-GameState.json'
    with open(anno_path, 'r') as f:
        json_data = json.load(f)

    labels = [0, 0]
    for anno in json_data['annotations']:
        try:
            if anno['category_id'] != 1:
                continue
            attributes = anno['attributes']
            if attributes['team'] == 'left' and labels[0] == 0:
                labels[0] += 1
            elif attributes['team'] == 'right' and labels[1] == 0:
                labels[1] += 1
            else:
                continue
            img_info = [x for x in json_data['images'] if x['image_id']==anno['image_id']][0]
            img = load_image(f"{dir_path}/img1/{img_info['file_name']}")
            bboxes = anno['bbox_image']
            x, y, w, h = bboxes['x'], bboxes['y'], bboxes['w'], bboxes['h']
            x1, y1, x2, y2 = [int(x) for x in [x, y, x + w, y + h]]
            croped = img.crop([x1, y1, x2, y2])
            croped.save(f"{OUTPUT_PATH}/{dir_name}_{attributes['team']}.jpg")
            if sum(labels) > 1:
                break
        except KeyError as e:
            print(e, anno_path)
