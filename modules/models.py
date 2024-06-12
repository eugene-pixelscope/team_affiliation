import torch
import torch.nn as nn
from .layers import BNClassifier, PixelToPartClassifier, GlobalMaskWeightedPoolingHead, GlobalPoolingHead
import torch.nn.functional as F
from collections import OrderedDict


class Network(nn.Module):
    def __init__(self, backbone, class_num):
        super(Network, self).__init__()

        self.backbone = backbone
        in_dim = self.backbone.rep_dim
        self.final_layer = nn.Linear(in_dim, class_num, bias=False)

    def forward(self, x):
        feature = self.backbone(x)
        output = self.final_layer(feature)
        return output


class PRTreIDTeamClassifier(nn.Module):
    """Team-Affiliation classifier in PRT-ReID.

    Reference:
        "Multi-task Learning for Joint Re-identification, Team Affiliation, and Role Classification for Sports Visual Tracking", doi:10.1145/3606038.3616172
    """
    def __init__(self, backbone, num_teams=2, num_role=2, attention_enable=False):
        super(PRTreIDTeamClassifier, self).__init__()
        self.backbone = backbone
        self.attention_enable = attention_enable

        in_dim = self.backbone.rep_dim
        self.flatten = nn.Flatten(1, 2)
        if self.attention_enable:
            self.attention = PixelToPartClassifier(in_dim)
            self.global_pooling = GlobalMaskWeightedPoolingHead()
        else:
            self.global_pooling = GlobalPoolingHead()
        self.team_classifier = BNClassifier(in_dim, num_teams)
        self.role_classifier = BNClassifier(in_dim, num_role)

    def forward(self, x):
        if self.attention_enable:
            return self._forward_attention(x)
        else:
            return self._forward(x)

    def _forward_attention(self, x):
        assert self.attention_enable

        spatial_features = self.backbone(x)
        pixels_cls_scores = self.attention(spatial_features)  # [N, K, Hf, Wf]
        masks = F.softmax(pixels_cls_scores, dim=1)[:, 0]

        embedding = self.global_pooling(spatial_features, masks.unsqueeze(1)).flatten(1, 2)  # [N, D]
        team_bn_embedding, team_cls_score = self.team_classifier(embedding)
        _, role_cls_score = self.self.role_classifier(embedding)

        feature = embedding if self.training else team_bn_embedding
        return feature, team_cls_score, F.sigmoid(pixels_cls_scores[:, 0]), role_cls_score

    def _forward(self, x):
        spatial_features = self.backbone(x)
        embedding = self.global_pooling(spatial_features).flatten(1, 2)  # [N, D]
        team_bn_embedding, team_cls_score = self.team_classifier(embedding)
        _, role_cls_score = self.self.role_classifier(embedding)

        feature = embedding if self.training else team_bn_embedding
        return feature, team_cls_score, role_cls_score


class TwoPhaseTeamClassifier(nn.Module):
    """Team-Affiliation classifier using color(yellow, green, blue, red, white).
    Output's shape is 5-dim vector and each dimension are consisted with the number of pixels of color.

    Reference:
        "T. Guo, K. Tao, Q. Hu and Y. Shen, "Detection of Ice Hockey Players and Teams via a Two-Phase Cascaded CNN Model", in IEEE Access, vol. 8, pp. 195062-195073, 2020, doi: 10.1109/ACCESS.2020.3033580.
    """
    def __init__(self):
        super(TwoPhaseTeamClassifier, self).__init__()
        self.net = self.color_split_torch
        # value in paper
        # self.color_range = {
        #     'yellow': [[15, 45], [51, 255], [30,215]],
        #     'green': [[45, 75], [51, 255], [30,215]],
        #     'blue': [[75, 135], [51, 255], [21,128]],
        #     'white': [[50, 165], [51, 255], [125,215]],
        #     'red': [[-15,15], [51,255], [30,215]],
        # }
        self.color_range = {
            'yellow': [[15, 45], [100, 255], [100,255]],
            'green': [[45, 75], [51, 255], [30,215]],
            'blue': [[75, 135], [51, 255], [21,128]],
            'white': [[50, 185], [0, 255], [155,255]],
            'red': [[-15,15], [51,255], [30,215]],
        }

    def color_split_torch(self, hsv, out):
        for i, (k, range) in enumerate(self.color_range.items()):
            if k == 'red':
                out[:,i] = (torch.logical_or(torch.logical_and(range[0][0] + 180 <= hsv[:,0,::], hsv[:,0,::] <= 180),
                                              torch.logical_and(0 <= hsv[:,0,::], hsv[:,0,::] <= range[0][1]))
                             * torch.logical_and(range[1][0] <= hsv[:,1,::], hsv[:,1,::] <= range[1][1])
                             * torch.logical_and(range[2][0] <= hsv[:,2,::], hsv[:,2,::] <= range[2][1])).to(dtype=torch.uint8)
            else:
                out[:,i] = (torch.logical_and(range[0][0] <= hsv[:,0,::], hsv[:,0,::] <= range[0][1])
                             * torch.logical_and(range[1][0] <= hsv[:,1,::], hsv[:,1,::] <= range[1][1])
                             * torch.logical_and(range[2][0] <= hsv[:,2,::], hsv[:,2,::] <= range[2][1])).to(dtype=torch.uint8)
        return out

    def forward(self, x):
        B, _, H, W = x.shape
        mask = torch.zeros(B, 5, H, W).to(x.get_device())
        c_k = self.net(x, mask)
        q_c_k = c_k.sum(dim=(2,3))
        T = torch.argmax(q_c_k, dim=1)
        return T, (c_k, q_c_k)

