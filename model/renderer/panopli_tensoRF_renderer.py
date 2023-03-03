# MIT License
#
# Copyright (c) 2022 Anpei Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import random
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch_efficient_distloss import eff_distloss

from util.distinct_colors import DistinctColors
from util.misc import visualize_points
from util.transforms import tr_comp, dot, trs_comp


class TensoRFRenderer(nn.Module):

    def __init__(self, bbox_aabb, grid_dim, stop_semantic_grad=True, semantic_weight_mode="none", step_ratio=0.5, distance_scale=25, raymarch_weight_thres=0.0001, alpha_mask_threshold=0.0075, parent_renderer_ref=None, instance_id=0):
        super().__init__()
        self.register_buffer("bbox_aabb", bbox_aabb)
        self.register_buffer("grid_dim", torch.LongTensor(grid_dim))
        self.register_buffer("inv_box_extent", torch.zeros([3]))
        self.register_buffer("units", torch.zeros([3]))
        self.semantic_weight_mode = semantic_weight_mode
        self.parent_renderer_ref = parent_renderer_ref
        self.step_ratio = step_ratio
        self.distance_scale = distance_scale
        self.raymarch_weight_thres = raymarch_weight_thres
        self.alpha_mask_threshold = alpha_mask_threshold
        self.step_size = None
        self.n_samples = None
        self.stop_semantic_grad = stop_semantic_grad
        self.instance_id = instance_id
        self.update_step_size(self.grid_dim)

    def update_step_size(self, grid_dim):
        print(f"\n[{self.instance_id:02d}] aabb", self.bbox_aabb.view(-1))
        print(f"[{self.instance_id:02d}] grid size", grid_dim)
        box_extent = self.bbox_aabb[1] - self.bbox_aabb[0]
        self.grid_dim.data = torch.tensor(grid_dim, device=self.bbox_aabb.device) if isinstance(grid_dim, tuple) else grid_dim
        self.inv_box_extent.data = 2.0 / box_extent
        self.units.data = box_extent / (self.grid_dim - 1 + 1e-3)
        print(f"[{self.instance_id:02d}] units: ", self.step_size)
        self.step_size = torch.mean(self.units) * self.step_ratio
        print(f"[{self.instance_id:02d}] sampling step size: ", self.step_size)
        box_diag = torch.sqrt(torch.sum(torch.square(box_extent)))
        self.n_samples = int((box_diag / self.step_size).item()) + 1
        print(f"[{self.instance_id:02d}] sampling number: ", self.n_samples)

    def update_step_ratio(self, step_ratio):
        self.step_ratio = step_ratio
        self.step_size = torch.mean(self.units) * self.step_ratio
        box_extent = self.bbox_aabb[1] - self.bbox_aabb[0]
        box_diag = torch.sqrt(torch.sum(torch.square(box_extent)))
        self.n_samples = int((box_diag / self.step_size).item()) + 1

    def forward(self, tensorf, rays, perturb, white_bg, is_train):
        xyz_sampled, z_vals, mask_xyz = sample_points_in_box(rays, self.bbox_aabb, self.n_samples, self.step_size, perturb, is_train)
        viewdirs = rays[:, 3:6].view(-1, 1, 3).expand(xyz_sampled.shape)
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        midpoints = torch.cat(((z_vals[:, 1:] + z_vals[:, :-1]) / 2, z_vals[:, -2:-1] * torch.ones_like(z_vals[:, :1])), dim=-1)
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        semantics = torch.zeros((*xyz_sampled.shape[:2], tensorf.num_semantic_classes), device=xyz_sampled.device)
        instances = torch.zeros((*xyz_sampled.shape[:2], tensorf.dim_feature_instance), device=xyz_sampled.device)
        # regfeats = torch.zeros((*xyz_sampled.shape[:2], 384), device=xyz_sampled.device)

        xyz_sampled = self.normalize_coordinates(xyz_sampled)
        if mask_xyz.any():
            sigma[mask_xyz] = tensorf.compute_density(xyz_sampled[mask_xyz])

        alpha, weight, bg_weight = self.raw_to_alpha(sigma, dists * self.distance_scale)
        dist_regularizer = eff_distloss(weight, midpoints, dists[:, :])

        appearance_mask = weight > self.raymarch_weight_thres

        if appearance_mask.any():
            appearance_features = tensorf.compute_appearance_feature(xyz_sampled[appearance_mask])
            valid_rgbs = tensorf.render_appearance_mlp(viewdirs[appearance_mask], appearance_features)
            rgb[appearance_mask] = valid_rgbs

            semantic_features = tensorf.compute_semantic_feature(xyz_sampled[appearance_mask])
            valid_semantics = tensorf.render_semantic_mlp(None, semantic_features)
            semantics[appearance_mask] = valid_semantics

            # debug table
            # sigma[torch.logical_and(appearance_mask, semantics.argmax(-1) == 7)] += 0.1

            instance_features = tensorf.compute_instance_feature(xyz_sampled[appearance_mask])
            instances[appearance_mask] = instance_features

            # if tensorf.use_feature_reg:
            #     regfeats[appearance_mask] = tensorf.render_semantic_mlp.get_backbone_feats(xyz_sampled[appearance_mask])
        # alpha, weight, bg_weight = self.raw_to_alpha(sigma, dists * self.distance_scale)

        opacity_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        regfeat_map = torch.zeros([1, 1], device=rgb_map.device)

        w = weight[..., None]
        if self.semantic_weight_mode == "argmax":
            w = torch.nn.functional.one_hot(w.argmax(dim=1)[:, 0], num_classes=w.shape[1]).unsqueeze(-1)
        if self.stop_semantic_grad:
            w = w.detach()
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)
            # if tensorf.use_feature_reg:
            #     regfeat_map = torch.sum(w * regfeats, -2)
        else:
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)
            # if tensorf.use_feature_reg:
            #     regfeat_map = torch.sum(w * regfeats, -2)

        if self.semantic_weight_mode == "softmax":
            semantic_map = semantic_map / (semantic_map.sum(-1).unsqueeze(-1) + 1e-8)
            semantic_map = torch.log(semantic_map + 1e-8)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - opacity_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)

        regfeat_map = torch.zeros([1, 1], device=rgb_map.device)
        return rgb_map, semantic_map, instance_map, depth_map, regfeat_map, dist_regularizer

    def forward_instance_feature(self, tensorf, rays, perturb, is_train):
        xyz_sampled, z_vals, mask_xyz = sample_points_in_box(rays, self.bbox_aabb, self.n_samples, self.step_size, perturb, is_train)
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        instances = torch.zeros((*xyz_sampled.shape[:2], tensorf.dim_feature_instance), device=xyz_sampled.device)

        xyz_sampled = self.normalize_coordinates(xyz_sampled)
        with torch.no_grad():
            if mask_xyz.any():
                sigma[mask_xyz] = tensorf.compute_density(xyz_sampled[mask_xyz])
            alpha, weight, bg_weight = self.raw_to_alpha(sigma, dists * self.distance_scale)

        appearance_mask = weight > self.raymarch_weight_thres

        if appearance_mask.any():
            instance_features = tensorf.compute_instance_feature(xyz_sampled[appearance_mask])
            instances[appearance_mask] = instance_features

        instance_map = torch.sum(weight[..., None] * instances, -2)
        return instance_map

    def forward_segment_feature(self, tensorf, rays, perturb, is_train):
        xyz_sampled, z_vals, mask_xyz = sample_points_in_box(rays, self.bbox_aabb, self.n_samples, self.step_size, perturb, is_train)
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        segments = torch.zeros((*xyz_sampled.shape[:2], tensorf.num_semantic_classes), device=xyz_sampled.device)

        xyz_sampled = self.normalize_coordinates(xyz_sampled)
        with torch.no_grad():
            if mask_xyz.any():
                sigma[mask_xyz] = tensorf.compute_density(xyz_sampled[mask_xyz])
            alpha, weight, bg_weight = self.raw_to_alpha(sigma, dists * self.distance_scale)

        appearance_mask = weight > self.raymarch_weight_thres

        if appearance_mask.any():
            segment_features = tensorf.compute_semantic_feature(xyz_sampled[appearance_mask])
            valid_semantics = tensorf.render_semantic_mlp(None, segment_features)
            segments[appearance_mask] = valid_semantics

        w = weight[..., None]
        w = w.detach()
        segment_map = torch.sum(w * segments, -2)

        if self.semantic_weight_mode == "softmax":
            segment_map = segment_map / (segment_map.sum(-1).unsqueeze(-1) + 1e-8)
            segment_map = torch.log(segment_map + 1e-8)

        return segment_map

    @torch.no_grad()
    def forward_delete(self, tensorf, rays, white_bg, bbox_deletion):
        xyz_sampled, z_vals, mask_xyz = sample_points_in_box(rays, self.bbox_aabb, self.n_samples, self.step_size, 0, False)
        _, delete_points = split_points_minimal(xyz_sampled.view(-1, 3), bbox_deletion["extent"].unsqueeze(0), bbox_deletion["position"].unsqueeze(0), bbox_deletion["orientation"].unsqueeze(0))
        delete_points = delete_points[0]

        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        semantics = torch.zeros((*xyz_sampled.shape[:2], tensorf.num_semantic_classes), device=xyz_sampled.device)
        instances = torch.zeros((*xyz_sampled.shape[:2], tensorf.dim_feature_instance), device=xyz_sampled.device)
        viewdirs = rays[:, 3:6].view(-1, 1, 3).expand(xyz_sampled.shape)
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        xyz_sampled = self.normalize_coordinates(xyz_sampled)
        if mask_xyz.any():
            sigma[mask_xyz] = tensorf.compute_density(xyz_sampled[mask_xyz])

        appearance_features = tensorf.compute_appearance_feature(xyz_sampled[mask_xyz])
        valid_rgbs = tensorf.render_appearance_mlp(viewdirs[mask_xyz], appearance_features)
        rgb[mask_xyz] = valid_rgbs

        semantic_features = tensorf.compute_semantic_feature(xyz_sampled[mask_xyz])
        valid_semantics = tensorf.render_semantic_mlp(None, semantic_features)
        semantics[mask_xyz] = valid_semantics

        instances[mask_xyz] = tensorf.compute_instance_feature(xyz_sampled[mask_xyz])

        sigma[delete_points.reshape(sigma.shape)] = 0

        alpha, weight, bg_weight = self.raw_to_alpha(sigma, dists * self.distance_scale)

        opacity_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        w = weight[..., None]
        if self.semantic_weight_mode == "argmax":
            w = torch.nn.functional.one_hot(w.argmax(dim=1)[:, 0], num_classes=w.shape[1]).unsqueeze(-1)
        if self.stop_semantic_grad:
            w = w.detach()
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)
        else:
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)

        if self.semantic_weight_mode == "softmax":
            semantic_map = semantic_map / (semantic_map.sum(-1).unsqueeze(-1) + 1e-8)
            semantic_map = torch.log(semantic_map + 1e-8)

        if white_bg:
            rgb_map = rgb_map + (1. - opacity_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)

        return rgb_map, semantic_map, instance_map, depth_map

    @torch.no_grad()
    def forward_extract(self, tensorf, rays, white_bg, bbox_extraction):
        xyz_sampled, z_vals, mask_xyz = sample_points_in_box(rays, self.bbox_aabb, self.n_samples, self.step_size, 0, False)
        _, extract_points = split_points_minimal(xyz_sampled.view(-1, 3), bbox_extraction["extent"].unsqueeze(0), bbox_extraction["position"].unsqueeze(0), bbox_extraction["orientation"].unsqueeze(0))
        extract_points = extract_points[0]
        delete_points = ~extract_points

        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        semantics = torch.zeros((*xyz_sampled.shape[:2], tensorf.num_semantic_classes), device=xyz_sampled.device)
        instances = torch.zeros((*xyz_sampled.shape[:2], tensorf.dim_feature_instance), device=xyz_sampled.device)
        viewdirs = rays[:, 3:6].view(-1, 1, 3).expand(xyz_sampled.shape)
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        xyz_sampled = self.normalize_coordinates(xyz_sampled)
        if mask_xyz.any():
            sigma[mask_xyz] = tensorf.compute_density(xyz_sampled[mask_xyz])

        appearance_features = tensorf.compute_appearance_feature(xyz_sampled[mask_xyz])
        valid_rgbs = tensorf.render_appearance_mlp(viewdirs[mask_xyz], appearance_features)
        rgb[mask_xyz] = valid_rgbs

        semantic_features = tensorf.compute_semantic_feature(xyz_sampled[mask_xyz])
        valid_semantics = tensorf.render_semantic_mlp(None, semantic_features)
        semantics[mask_xyz] = valid_semantics

        instances[mask_xyz] = tensorf.compute_instance_feature(xyz_sampled[mask_xyz])

        sigma[delete_points.reshape(sigma.shape)] = 0

        alpha, weight, bg_weight = self.raw_to_alpha(sigma, dists * self.distance_scale)

        opacity_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        w = weight[..., None]
        if self.semantic_weight_mode == "argmax":
            w = torch.nn.functional.one_hot(w.argmax(dim=1)[:, 0], num_classes=w.shape[1]).unsqueeze(-1)
        if self.stop_semantic_grad:
            w = w.detach()
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)
        else:
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)

        if self.semantic_weight_mode == "softmax":
            semantic_map = semantic_map / (semantic_map.sum(-1).unsqueeze(-1) + 1e-8)
            semantic_map = torch.log(semantic_map + 1e-8)

        if white_bg:
            rgb_map = rgb_map + (1. - opacity_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)

        return rgb_map, semantic_map, instance_map, depth_map

    @torch.no_grad()
    def forward_duplicate(self, tensorf, rays, white_bg, bbox_instance, translation, rotation):
        xyz_sampled, z_vals, mask_xyz = sample_points_in_box(rays, self.bbox_aabb, self.n_samples, self.step_size, 0, False)
        _, manipulated_points = split_points_minimal(xyz_sampled.view(-1, 3), bbox_instance["extent"].unsqueeze(0), (rotation @ bbox_instance["position"] + translation).unsqueeze(0), (rotation @ bbox_instance["orientation"]).unsqueeze(0))
        manipulated_points = manipulated_points[0]
        orig_dim_xyz = xyz_sampled.shape
        xyz_sampled = xyz_sampled.reshape(-1, 3)
        xyz_sampled[manipulated_points, :] = dot(torch.linalg.inv(tr_comp(translation, torch.eye(3).cuda())), xyz_sampled[manipulated_points, :])
        xyz_sampled = xyz_sampled.reshape(orig_dim_xyz)

        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        semantics = torch.zeros((*xyz_sampled.shape[:2], tensorf.num_semantic_classes), device=xyz_sampled.device)
        instances = torch.zeros((*xyz_sampled.shape[:2], tensorf.dim_feature_instance), device=xyz_sampled.device)
        viewdirs = rays[:, 3:6].view(-1, 1, 3).expand(xyz_sampled.shape)
        orig_dim_viewdirs = viewdirs.shape
        viewdirs = viewdirs.reshape(-1, orig_dim_viewdirs[-1])
        viewdirs[manipulated_points, :] = (torch.linalg.inv(rotation) @ viewdirs[manipulated_points, :].T).T
        viewdirs = viewdirs.reshape(orig_dim_viewdirs)

        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        xyz_sampled = self.normalize_coordinates(xyz_sampled)
        if mask_xyz.any():
            sigma[mask_xyz] = tensorf.compute_density(xyz_sampled[mask_xyz])

        appearance_features = tensorf.compute_appearance_feature(xyz_sampled[mask_xyz])
        valid_rgbs = tensorf.render_appearance_mlp(viewdirs[mask_xyz], appearance_features)
        rgb[mask_xyz] = valid_rgbs

        semantic_features = tensorf.compute_semantic_feature(xyz_sampled[mask_xyz])
        valid_semantics = tensorf.render_semantic_mlp(None, semantic_features)
        semantics[mask_xyz] = valid_semantics

        instances[mask_xyz] = tensorf.compute_instance_feature(xyz_sampled[mask_xyz])

        alpha, weight, bg_weight = self.raw_to_alpha(sigma, dists * self.distance_scale)

        opacity_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        w = weight[..., None]
        if self.semantic_weight_mode == "argmax":
            w = torch.nn.functional.one_hot(w.argmax(dim=1)[:, 0], num_classes=w.shape[1]).unsqueeze(-1)
        if self.stop_semantic_grad:
            w = w.detach()
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)
        else:
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)

        if self.semantic_weight_mode == "softmax":
            semantic_map = semantic_map / (semantic_map.sum(-1).unsqueeze(-1) + 1e-8)
            semantic_map = torch.log(semantic_map + 1e-8)

        if white_bg:
            rgb_map = rgb_map + (1. - opacity_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)

        return rgb_map, semantic_map, instance_map, depth_map

    @torch.no_grad()
    def forward_manipulate(self, tensorf, rays, white_bg, bbox_instance, translation, rotation):
        xyz_sampled, z_vals, mask_xyz = sample_points_in_box(rays, self.bbox_aabb, self.n_samples, self.step_size, 0, False)
        _, manipulated_points = split_points_minimal(xyz_sampled.view(-1, 3), bbox_instance["extent"].unsqueeze(0), (bbox_instance["position"] + translation).unsqueeze(0), (rotation @ bbox_instance["orientation"]).unsqueeze(0))
        manipulated_points = manipulated_points[0]
        _, bbox_points = split_points_minimal(xyz_sampled.view(-1, 3), bbox_instance["extent"].unsqueeze(0), bbox_instance["position"].unsqueeze(0), bbox_instance["orientation"].unsqueeze(0))
        bbox_points = bbox_points[0]
        orig_dim_xyz = xyz_sampled.shape
        xyz_sampled = xyz_sampled.reshape(-1, 3)

        xyz_sampled[manipulated_points, :] = (rotation @ (xyz_sampled[manipulated_points, :] - bbox_instance["position"]).T).T + bbox_instance["position"] - translation
        xyz_sampled = xyz_sampled.reshape(orig_dim_xyz)

        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        semantics = torch.zeros((*xyz_sampled.shape[:2], tensorf.num_semantic_classes), device=xyz_sampled.device)
        instances = torch.zeros((*xyz_sampled.shape[:2], tensorf.dim_feature_instance), device=xyz_sampled.device)
        viewdirs = rays[:, 3:6].view(-1, 1, 3).expand(xyz_sampled.shape)
        orig_dim_viewdirs = viewdirs.shape
        viewdirs = viewdirs.reshape(-1, orig_dim_viewdirs[-1])
        viewdirs[manipulated_points, :] = (torch.linalg.inv(rotation) @ viewdirs[manipulated_points, :].T).T
        viewdirs = viewdirs.reshape(orig_dim_viewdirs)

        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        xyz_sampled = self.normalize_coordinates(xyz_sampled)
        if mask_xyz.any():
            sigma[mask_xyz] = tensorf.compute_density(xyz_sampled[mask_xyz])

        appearance_features = tensorf.compute_appearance_feature(xyz_sampled[mask_xyz])
        valid_rgbs = tensorf.render_appearance_mlp(viewdirs[mask_xyz], appearance_features)
        rgb[mask_xyz] = valid_rgbs

        semantic_features = tensorf.compute_semantic_feature(xyz_sampled[mask_xyz])
        valid_semantics = tensorf.render_semantic_mlp(None, semantic_features)
        semantics[mask_xyz] = valid_semantics

        instances[mask_xyz] = tensorf.compute_instance_feature(xyz_sampled[mask_xyz])

        sigma[torch.logical_and(bbox_points, ~manipulated_points).reshape(sigma.shape)] = 0
        alpha, weight, bg_weight = self.raw_to_alpha(sigma, dists * self.distance_scale)

        opacity_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        w = weight[..., None]
        if self.semantic_weight_mode == "argmax":
            w = torch.nn.functional.one_hot(w.argmax(dim=1)[:, 0], num_classes=w.shape[1]).unsqueeze(-1)
        if self.stop_semantic_grad:
            w = w.detach()
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)
        else:
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)

        if self.semantic_weight_mode == "softmax":
            semantic_map = semantic_map / (semantic_map.sum(-1).unsqueeze(-1) + 1e-8)
            semantic_map = torch.log(semantic_map + 1e-8)

        if white_bg:
            rgb_map = rgb_map + (1. - opacity_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)

        return rgb_map, semantic_map, instance_map, depth_map


    @staticmethod
    def raw_to_alpha(sigma, dist):
        alpha = 1. - torch.exp(-sigma * dist)
        T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
        weights = alpha * T[:, :-1]
        return alpha, weights, T[:, -1:]

    def normalize_coordinates(self, xyz_sampled):
        return (xyz_sampled - self.bbox_aabb[0]) * self.inv_box_extent - 1

    @torch.no_grad()
    def get_instance_clusters(self, tensorf, mode):
        alpha, dense_xyz = self.get_dense_alpha(tensorf)
        xyz_sampled = self.normalize_coordinates(dense_xyz)
        labels = tensorf.compute_instance_feature(xyz_sampled.view(-1, 3)).reshape([xyz_sampled.shape[0], xyz_sampled.shape[1], xyz_sampled.shape[2], -1])
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        labels = labels.transpose(0, 2).contiguous().view([xyz_sampled.shape[0] * xyz_sampled.shape[1] * xyz_sampled.shape[2], -1]).argmax(dim=1).int()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()
        alpha[alpha >= self.alpha_mask_threshold] = 1
        alpha[alpha < self.alpha_mask_threshold] = 0
        if mode == 'full':
            max_samples = 2 ** 16
            valid_xyz = dense_xyz[alpha >= 0]
        else:
            max_samples = 2 ** 18
            mask = alpha > 0.5
            valid_xyz = dense_xyz[mask]
            labels = labels[mask.view(-1)]
        selected_indices = random.sample(list(range(valid_xyz.shape[0])), min(max_samples, valid_xyz.shape[0]))
        valid_xyz = valid_xyz[selected_indices, :]
        valid_labels = labels[selected_indices]
        return valid_xyz, valid_labels

    @torch.no_grad()
    def update_bbox_aabb_and_shrink(self, tensorf, fractional_lenience=1.0):
        alpha, dense_xyz = self.get_dense_alpha(tensorf)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]

        alpha = F.max_pool3d(alpha, kernel_size=3, padding=1, stride=1).view(self.grid_dim.tolist()[::-1])
        alpha[alpha >= self.alpha_mask_threshold] = 1
        alpha[alpha < self.alpha_mask_threshold] = 0

        valid_xyz = dense_xyz[alpha > 0.5]

        if valid_xyz.shape[0] > 0:
            xyz_min = valid_xyz.amin(0)
            xyz_max = valid_xyz.amax(0)

            # adjust xyz_min, xyz_max based on leniency factor
            extent = xyz_max - xyz_min
            position = (xyz_min + xyz_max) / 2
            xyz_min_fl = position - (extent * fractional_lenience) / 2
            xyz_max_fl = position + (extent * fractional_lenience) / 2

            box_min, box_max = self.bbox_aabb[0], self.bbox_aabb[1]
            xyz_min = torch.maximum(box_min, xyz_min_fl)
            xyz_max = torch.minimum(box_max, xyz_max_fl)

            if self.parent_renderer_ref is not None:
                box_min, box_max = self.parent_renderer_ref.bbox_aabb[0], self.parent_renderer_ref.bbox_aabb[1]
                xyz_min = torch.maximum(box_min, xyz_min)
                xyz_max = torch.minimum(box_max, xyz_max)

            new_bbox_aabb = torch.stack((xyz_min, xyz_max))

            total = torch.sum(alpha)
            print(f"[{self.instance_id:02d}] bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100))
            xyz_min, xyz_max = new_bbox_aabb
            t_l, b_r = (xyz_min - self.bbox_aabb[0]) / self.units, (xyz_max - self.bbox_aabb[0]) / self.units
            t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
            b_r = torch.stack([b_r, self.grid_dim]).amin(0)
            new_size = b_r - t_l
            if new_size[0] > 0 and new_size[1] > 0 and new_size[2] > 0:
                print(f"[{self.instance_id:02d}] shrinking ... with grid_size {new_size}")
                tensorf.shrink(t_l, b_r)
                self.bbox_aabb.data = new_bbox_aabb
                self.update_step_size((new_size[0], new_size[1], new_size[2]))
        else:
            print(f"[{self.instance_id:02d}] no valid voxels found ...")

    @torch.no_grad()
    def get_dense_alpha(self, tensorf):
        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, self.grid_dim[0]),
            torch.linspace(0, 1, self.grid_dim[1]),
            torch.linspace(0, 1, self.grid_dim[2]),
            indexing='ij'
        ), -1).to(tensorf.density_line[0].device)
        dense_xyz = self.bbox_aabb[0] * (1 - samples) + self.bbox_aabb[1] * samples
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(self.grid_dim[0]):
            alpha[i] = self.compute_alpha(tensorf, dense_xyz[i].view(-1, 3), self.step_size).view((self.grid_dim[1], self.grid_dim[2]))
        return alpha, dense_xyz

    def compute_sigma(self, tensorf, xyz_locs):
        xyz_sampled = self.normalize_coordinates(xyz_locs)
        sigma = tensorf.compute_density(xyz_sampled.view(-1, 3)).reshape(xyz_locs.shape[:-1])
        return sigma

    @torch.no_grad()
    def get_dense_sigma(self, tensorf, upsample=1):
        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, self.grid_dim[0] * upsample, device="cpu"),
            torch.linspace(0, 1, self.grid_dim[1] * upsample, device="cpu"),
            torch.linspace(0, 1, self.grid_dim[2] * upsample, device="cpu"),
            indexing='ij'
        ), -1)
        sigma = torch.zeros_like(samples[..., 0]).to(self.bbox_aabb.device)
        for i in tqdm(range(self.grid_dim[0] * upsample)):
            dense_xyz = self.bbox_aabb[0].cpu() * (1 - samples[i]) + self.bbox_aabb[1].cpu() * samples[i]
            sigma[i] = self.compute_sigma(tensorf, dense_xyz.view(-1, 3).to(self.bbox_aabb.device)).view((self.grid_dim[1] * upsample, self.grid_dim[2] * upsample))
        return sigma

    def compute_alpha(self, tensorf, xyz_locs, step_size):
        xyz_sampled = self.normalize_coordinates(xyz_locs)
        sigma = tensorf.compute_density(xyz_sampled.view(-1, 3)).reshape(xyz_locs.shape[:-1])
        alpha = 1 - torch.exp(-sigma * step_size).view(xyz_locs.shape[:-1])
        return alpha

    def get_target_resolution(self, n_voxels):
        xyz_min, xyz_max = self.bbox_aabb
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
        target_res = ((xyz_max - xyz_min) / voxel_size).long().tolist()
        target_res = [max(x, 1) for x in target_res]
        return tuple(target_res)

    @property
    def extent(self):
        return self.bbox_aabb[1] - self.bbox_aabb[0]

    @property
    def position(self):
        return (self.bbox_aabb[0] + self.bbox_aabb[1]) / 2

    @property
    def orientation(self):
        return torch.eye(3, device=self.bbox_aabb.device)

    def export_instance_clusters(self, tensorf, output_directory):
        color_manager = DistinctColors()
        c_xyz, c_label = self.get_instance_clusters(tensorf, mode='alpha')
        colors = color_manager.apply_colors_fast_torch(c_label.cpu().long())
        visualize_points(c_xyz.cpu().numpy(), output_directory / f"alpha.obj", colors=colors.numpy())
        c_xyz, c_label = self.get_instance_clusters(tensorf, mode='full')
        colors = color_manager.apply_colors_fast_torch(c_label.cpu().long())
        visualize_points(c_xyz.cpu().numpy(), output_directory / f"full.obj", colors=colors.numpy())


def split_points_minimal(xyz, extents, positions, orientations):
    split_xyz = []
    point_flags = []
    for i in range(extents.shape[0]):
        inverse_transform = torch.linalg.inv(trs_comp(positions[i], orientations[i], torch.ones([1], device=xyz.device)))
        inverse_transformed_xyz = (inverse_transform @ torch.cat([xyz, torch.ones([xyz.shape[0], 1], device=xyz.device)], 1).T).T[:, :3]
        t0 = torch.logical_and(inverse_transformed_xyz[:, 0] <= extents[i, 0] / 2, inverse_transformed_xyz[:, 0] >= -extents[i, 0] / 2)
        t1 = torch.logical_and(inverse_transformed_xyz[:, 1] <= extents[i, 1] / 2, inverse_transformed_xyz[:, 1] >= -extents[i, 1] / 2)
        t2 = torch.logical_and(inverse_transformed_xyz[:, 2] <= extents[i, 2] / 2, inverse_transformed_xyz[:, 2] >= -extents[i, 2] / 2)
        selection = torch.logical_and(torch.logical_and(t0, t1), t2)
        point_flags.append(selection)
        split_xyz.append(xyz[selection, :])
    return split_xyz, point_flags


def sample_points_in_box(rays, bbox_aabb, n_samples, step_size, perturb, is_train):
    rays_o, rays_d, nears, fars = rays[:, 0:3], rays[:, 3:6], rays[:, 6], rays[:, 7]
    vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
    rate_a = (bbox_aabb[1] - rays_o) / vec
    rate_b = (bbox_aabb[0] - rays_o) / vec
    t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=nears, max=fars)

    rng = torch.arange(n_samples)[None].float()
    if is_train and perturb != 0:
        rng = rng.repeat(rays_d.shape[-2], 1)
        rng = rng + perturb * torch.rand_like(rng[:, [0]])
    step = step_size * rng.to(rays_o.device)
    interpx = (t_min[..., None] + step)

    rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
    mask_outbbox = ((bbox_aabb[0] > rays_pts) | (rays_pts > bbox_aabb[1])).any(dim=-1)

    return rays_pts, interpx, ~mask_outbbox
