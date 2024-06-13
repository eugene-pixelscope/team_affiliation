from glob import glob
import os
from utils.vis_utils import show_mask_on_image

from modules import resnet
from modules.models import PRTreIDTeamClassifier
from utils.model_utils import load_model
import torch.nn.functional as F

import torch
import cv2
import numpy as np
from fast_pytorch_kmeans import KMeans


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256))
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    result = torch.tensor(img, dtype=torch.float) / 255
    return result

seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
backbone = resnet.get_resnet('ResNet18')
model_path = 'weights/best_with_role.tar'
# model_path = 'weights/best.tar'
model = PRTreIDTeamClassifier(backbone=backbone, num_teams=12, attention_enable=True)
model = model.to(device)

checkpoint = torch.load(model_path)
model = load_model(model, checkpoint, filter_team_classifier=True)
model.eval()

# setting input
def set_input(dir_path):
    inputs = []
    labels = []
    pathes = glob(os.path.join(dir_path, '*.png'))
    pathes.sort(key=lambda x: int(os.path.basename(x).split('-')[-1].replace('.png', '')))

    for k, path in enumerate(pathes):
        img = cv2.imread(path)
        labels.append(['1', '2', '3'].index(os.path.basename(path).replace('.png','').split('-')[-1]))
        img = preprocess(img).to(device).float()
        inputs.append(img)
    return inputs, labels

inputs, labels = set_input('demo_image')
pred_labels = []
feature_vector = []
for i, input_tensor in enumerate(inputs):
    with torch.no_grad():
        f, _, mask, role_cls = model(input_tensor)
    # visualization
    role_cls_softmax = F.softmax(role_cls, dim=1)
    role = torch.argmax(role_cls_softmax, dim=1)
    vis_img = show_mask_on_image(img=input_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),
                                 mask=mask.squeeze(0).cpu().detach().numpy(),
                                 image_weight=.7)
    cv2.imwrite(f'output/{i}_map.jpg', vis_img)

    if not bool(role) or bool(role_cls_softmax.squeeze(0)[role] < 0.7):
        feature_vector.append(f.squeeze(0))
        pred_labels.append(0)
    else:
        pred_labels.append(2)
feature_vector = torch.stack(feature_vector)
# kmeans
kmeans = KMeans(n_clusters=2, mode='cosine', max_iter=100)
c = kmeans.fit_predict(feature_vector)

pred = c.detach().cpu().numpy().tolist()
c_idx = 0
for i in range(len(pred_labels)):
    if pred_labels[i] == 0:
        pred_labels[i] += pred[c_idx]
        c_idx += 1
print('pred: ', pred_labels)
print('gt: ', labels)
print('Demo End')
