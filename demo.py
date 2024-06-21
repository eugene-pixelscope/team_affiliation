from glob import glob
import os
from utils.vis_utils import show_mask_on_image

from modules import resnet, shufflenetv2, efficientnet
from modules.models import PRTreIDTeamClassifier
from utils.model_utils import load_model
import torch.nn.functional as F

import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import CLARA
import matplotlib.pyplot as plt
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import seaborn as sns
from sklearn import preprocessing

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256))
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    result = torch.tensor(img, dtype=torch.float) / 255
    return result

# hyper param
role_thr = 0.6

# **********************
# 0. Initialize setting
# **********************
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# **********************
# 1. Model construct
# **********************
model_path = 'save/Hockey/prtrei_resnet18_checkpoint_60_with_role.tar'
if 'resnet' in os.path.basename(model_path):
    backbone = resnet.get_resnet('ResNet18')
elif 'shufflenetv2' in os.path.basename(model_path):
    backbone = shufflenetv2.get_shufflenet('ShuffleNetV2')
elif 'efficientnet' in os.path.basename(model_path):
    backbone = efficientnet.get_efficientnet('EfficientNetB0')
else:
    raise NotImplementedError

model = PRTreIDTeamClassifier(backbone=backbone, num_teams=12, attention_enable=True)
model = model.to(device)

checkpoint = torch.load(model_path, map_location=device)
load_model(model, checkpoint, filter_team_classifier=True)
model.eval()

# setting input
def set_input(dir_path):
    std_labels = ['1', '2']
    inputs = []
    labels = []
    pathes = glob(os.path.join(dir_path, '*.png'))
    pathes.sort(key=lambda x: int(os.path.basename(x).split('-')[-1].replace('.png', '')))

    for k, path in enumerate(pathes):
        img = cv2.imread(path)
        c = os.path.basename(path).replace('.png','').split('-')[-1]
        if c not in std_labels:
            continue
        labels.append(std_labels.index(c))
        # **********************
        # 2. Preprocessing
        # **********************
        img = preprocess(img).to(device).float()
        inputs.append(img)
    return inputs, labels

inputs, labels = set_input('demo_image')
pred_labels = []
feature_vector = []
# **********************
# 3. Inference
# **********************
for i, input_tensor in enumerate(inputs):
    with torch.no_grad():
        f, _, mask, role_cls = model(input_tensor)
    # visualization
    role = torch.argmax(role_cls, dim=1)
    vis_img = show_mask_on_image(img=input_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),
                                 mask=mask.squeeze(0).cpu().detach().numpy(),
                                 image_weight=.7)
    cv2.imwrite(f'output/{i}_map.jpg', vis_img)
    feature_vector.append(f.squeeze(0))
    pred_labels.append(0)

# **********************
# 4. Postprocessing
# **********************
feature_vector = torch.stack(feature_vector)
feature = feature_vector.detach().cpu().numpy()
nor_feature = preprocessing.normalize(feature)

# **********************
# 5. Clustering
# **********************
# clustering_engine = KMeans(n_clusters=2, random_state=seed, max_iter=20, n_init=10)
clustering_engine = CLARA(n_clusters=2, random_state=seed, max_iter=20)
clustering_engine.fit(nor_feature)
pred = clustering_engine.labels_

out = pairwise_cosine_similarity(feature_vector)
out.fill_diagonal_(1)
plt.clf()
ax = sns.heatmap(out.detach().cpu().numpy(), linewidth=0.5, annot=True, fmt=".2f")
plt.savefig(f'output/cosine.png')

c_idx = 0
for i in range(len(pred_labels)):
    if pred_labels[i] == 0:
        pred_labels[i] += pred[c_idx]
        c_idx += 1
print('pred: ', pred_labels)
print('gt: ', labels)
print('Demo End')
