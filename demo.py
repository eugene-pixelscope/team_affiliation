import os
import cv2
from modules import transform, resnet
from modules.models import PRTreIDTeamClassifier
import torch
import numpy as np
import os
from glob import glob
from kmeans_pytorch import kmeans
from torchvision.transforms import Normalize
from utils.vis_utils import show_mask_on_image
np.random.seed(123)

def preprocess(img, mean=None, std=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,256))
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    result = torch.tensor(img, dtype=torch.float) / 255
    if mean is not None and std is not None:
        result = Normalize(mean=mean, std=std)(result)
    return result


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
backbone = resnet.get_resnet('ResNet34')
# model_path = 'save/IceHockey/prtrei_resnet18_checkpoint_80.tar'
model_path = 'save/SoccerNetGS/prtrei_resnet18_checkpoint_80.tar'
attention_enable = True
model = PRTreIDTeamClassifier(backbone=backbone, num_teams=12, attention_enable=attention_enable)
model = model.to(device)


checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['net'], strict=False)
model.eval()

# setting input
def set_input(dir_path):
    inputs = []
    labels = []
    pathes = glob(os.path.join(dir_path, '*.png'))
    pathes.sort()

    for k, path in enumerate(pathes):
        img = cv2.imread(path)
        labels.append(['l', 'r'].index(os.path.basename(path).split('_')[0]))
        img = preprocess(img).to(device).float()
        # img = preprocess(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device).float()
        inputs.append(img)
    return inputs, labels

inputs, labels = set_input('demo_image/frame_1')
print(labels)

feature_vector = []
for i, vector in enumerate(inputs):
    with torch.no_grad():
        if attention_enable:
            f, _, mask = model(vector)
            B, C, _, _ = vector.shape
            # visualization
            for b in range(B):
                vis_img = show_mask_on_image(img=vector[b].permute(1, 2, 0).cpu().detach().numpy(), mask=mask[b].cpu().detach().numpy(), image_weight=.7)
                cv2.imwrite(f'output/{i}_{b}_map.jpg', vis_img)
        else:
            f = model.forward(vector)
    f = f.detach().cpu().numpy()
    feature_vector.extend(f)

feature_vector = torch.from_numpy(np.array(feature_vector))
cluster_ids_x, cluster_centers = kmeans(
    X=feature_vector, num_clusters=2, distance='cosine', device=device
)

print('dnn+kmeans: ', cluster_ids_x)

print('Demo End')
