from glob import glob
import os
from utils.vis_utils import show_mask_on_image

from modules import resnet
from modules.models import PRTreIDTeamClassifier
from utils.model_utils import load_model

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

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
backbone = resnet.get_resnet('ResNet18')
model_path = 'weights/best.tar'
model = PRTreIDTeamClassifier(backbone=backbone, num_teams=12, attention_enable=True)
model = model.to(device)

checkpoint = torch.load(model_path)
model = load_model(model, checkpoint)
model.eval()

# setting input
def set_input(dir_path):
    inputs = []
    labels = []
    pathes = glob(os.path.join(dir_path, '*.png'))
    pathes.sort()

    for k, path in enumerate(pathes):
        img = cv2.imread(path)
        labels.append(['1', '2'].index(os.path.basename(path).replace('.png','').split('-')[-1]))
        img = preprocess(img).to(device).float()
        inputs.append(img)
    return inputs, labels

inputs, labels = set_input('demo_image')

feature_vector = []
for i, input_tensor in enumerate(inputs):
    with torch.no_grad():
        f, _, mask = model(input_tensor)
        # visualization
        vis_img = show_mask_on_image(img=input_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),
                                     mask=mask.squeeze(0).cpu().detach().numpy(),
                                     image_weight=.7)
        cv2.imwrite(f'output/{i}_map.jpg', vis_img)
    feature_vector.append(f.squeeze(0))
feature_vector = torch.stack(feature_vector)
# kmeans
kmeans = KMeans(n_clusters=2, mode='euclidean', verbose=0, max_iter=20)
c = kmeans.fit_predict(feature_vector)

pred = c.detach().cpu().numpy().tolist()
print('pred: ', pred)
print('gt: ', labels)
print('Demo End')
