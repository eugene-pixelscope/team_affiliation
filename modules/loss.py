import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        # self.cosine_distance = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cosine_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)

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
    # def compute_dist_matrix(self, inputs):
    #     return 1 - self.cosine_distance(inputs, inputs)

    def compute_dist_matrix(self, inputs):
        # Compute cosine similarity and convert to distance
        cosine_sim = self.cosine_similarity(inputs.unsqueeze(1), inputs.unsqueeze(0))
        dist = 1 - cosine_sim
        return dist

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


class CrossEntropyLoss(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by

    .. math::
        \begin{equation}
        (1 - \eps) \times y + \frac{\eps}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\eps` is a weight. When
    :math:`\eps = 0`, the loss function reduces to the normal cross entropy.

    Args:
        num_classes (int): number of classes.
        eps (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, eps=0.1, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.eps = eps if label_smooth else 0
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, weights=None):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        assert inputs.shape[0] == targets.shape[0]
        num_classes = inputs.shape[1]
        log_probs = self.logsoftmax(inputs)
        zeros = torch.zeros(log_probs.size())
        targets = zeros.scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if inputs.is_cuda:
            targets = targets.cuda()
        targets = (1 - self.eps) * targets + self.eps / num_classes
        if weights is not None:
            result = (-targets * log_probs).sum(dim=1)
            result = result * nn.functional.normalize(weights, p=1, dim=0)
            result = result.sum()
        else:
            result = (-targets * log_probs).mean(0).sum()
        return result
