# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import numpy as np


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr_hierarchy(image_pred, image_gt, valid_mask=None, reduction='mean'):
    psnr_coarse, psnr_fine = 100, 100
    if 'rgb_coarse' in image_pred:
        psnr_coarse = -10*torch.log10(mse(image_pred['rgb_coarse'].detach(), image_gt, valid_mask, reduction))
    if 'rgb_fine' in image_pred:
        psnr_fine = -10 * torch.log10(mse(image_pred['rgb_fine'].detach(), image_gt, valid_mask, reduction))
    return psnr_coarse, psnr_fine


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred.detach(), image_gt, valid_mask, reduction))


def get_non_robust_classes(confusion_matrix, robustness_thres):
    axis_0 = np.sum(confusion_matrix, axis=0)
    axis_1 = np.sum(confusion_matrix, axis=1)
    total_labels = axis_0.sum()
    non_robust_0 = axis_0 / total_labels < robustness_thres
    non_robust_1 = axis_1 / total_labels < robustness_thres
    return np.where(np.logical_and(non_robust_0, non_robust_1))[0].tolist()


def calculate_miou(confusion_matrix, ignore_class=None, robust=0.005):
    MIoU = np.divide(np.diag(confusion_matrix), (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)))
    if ignore_class is not None:
        ignore_class += get_non_robust_classes(confusion_matrix, robust)
        for i in ignore_class:
            MIoU[i] = float('nan')
    MIoU = np.nanmean(MIoU)
    return MIoU


class ConfusionMatrix:

    def __init__(self, num_classes, ignore_class=None, robust=0.005):
        np.seterr(divide='ignore', invalid='ignore')
        self.num_class = num_classes
        self.ignore_class = ignore_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.robust = robust

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image, return_miou=False):
        assert gt_image.shape == pre_image.shape
        confusion_matrix = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += confusion_matrix
        if return_miou:
            return calculate_miou(confusion_matrix, self.ignore_class, self.robust)

    def get_miou(self):
        return calculate_miou(self.confusion_matrix, self.ignore_class, self.robust)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
