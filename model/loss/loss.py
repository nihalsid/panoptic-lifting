# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from torch import nn
import torch.nn.functional as F


class NeRFMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, inputs, targets):
        loss_coarse = torch.zeros([1], device=targets.device)
        loss_fine = torch.zeros([1], device=targets.device)
        if 'rgb_coarse' in inputs:
            loss_coarse = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss_fine = self.loss(inputs['rgb_fine'], targets)
        return loss_coarse, loss_fine


class NeRFSemanticsLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets, key):
        loss_coarse = torch.zeros([1], device=targets.device)
        loss_fine = torch.zeros([1], device=targets.device)
        if f'{key}_coarse' in inputs:
            loss_coarse = self.loss(inputs[f'{key}_coarse'], targets)
        if f'{key}_fine' in inputs:
            loss_fine = self.loss(inputs[f'{key}_fine'], targets)
        return loss_coarse, loss_fine


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.size_tensor(x[:, :, 1:, :]) + 1e-4
        count_w = self.size_tensor(x[:, :, :, 1:]) + 1e-4
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def size_tensor(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def get_semantic_weights(reweight_classes, fg_classes, num_semantic_classes):
    weights = torch.ones([num_semantic_classes]).float()
    if reweight_classes:
        weights[fg_classes] = 2
    return weights


class InstanceMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, rgb_all_instance, targets, instances):
        targets = targets.unsqueeze(1).expand(-1, rgb_all_instance.shape[-1], -1)
        instances = instances.unsqueeze(-1).expand(-1, rgb_all_instance.shape[-1])
        instance_values = torch.tensor(list(range(rgb_all_instance.shape[-1]))).to(instances.device).unsqueeze(0).expand(instances.shape[0], -1)
        instance_mask = instances == instance_values
        loss = self.loss(rgb_all_instance.permute((0, 2, 1)).reshape(-1, 3)[instance_mask.view(-1), :], targets.reshape(-1, 3)[instance_mask.view(-1), :])
        return loss


class MaskedNLLLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss(reduction='mean')

    def forward(self, output_instances, instances, semantics, invalid_class):
        if invalid_class is None:
            return self.loss(output_instances, instances)
        mask = semantics != invalid_class
        return self.loss(output_instances[mask, :], instances[mask])


class SCELoss(torch.nn.Module):

    def __init__(self, alpha, beta, class_weights):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, pred, labels_probabilities):
        # CCE
        ce = self.cross_entropy(pred, labels_probabilities)

        # RCE
        weights = torch.tensor(self.class_weights, device=pred.device).unsqueeze(0)
        pred = F.softmax(pred * weights, dim=1)
        pred = torch.clamp(pred, min=1e-8, max=1.0)
        label_clipped = torch.clamp(labels_probabilities, min=1e-8, max=1.0)

        rce = torch.sum(-1 * (pred * torch.log(label_clipped) * weights), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss
