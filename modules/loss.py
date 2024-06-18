import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.cosine_distance = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """

        # Compute pairwise distance
        dist = self.compute_dist_matrix(inputs)

        # For each anchor, find the hardest positive and negative
        return self.compute_hard_mine_triplet_loss(dist, targets)

    def compute_hard_mine_triplet_loss(self, dist, targets):
        n = dist.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # Use masked_select to find hardest positive/negative pairs
        dist_ap = torch.max(dist * mask.float(), dim=1)[0]
        dist_an = torch.min(dist * (1 - mask.float()) + mask.float() * 1e12, dim=1)[0]

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

    # def compute_dist_matrix(self, inputs):
    #     return torch.cdist(inputs, inputs, p=2)
    def compute_dist_matrix(self, inputs):
        return 1 - self.cosine_distance(inputs, inputs)

class AttentionLoss(nn.Module):

    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.cross_entropy = CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, attentions, masks):
        """ Compute loss for body part attention prediction.
            Args:
                attentions [N, H, W]
                masks [N, H, W]
            Returns:
        """
        gt_attentions = masks.flatten(1, 2)  # [N*Hf*Wf]
        attentions = attentions.flatten(1, 2)  # [N*Hf*Wf, M]
        loss = self.cross_entropy(attentions, gt_attentions)
        return loss
