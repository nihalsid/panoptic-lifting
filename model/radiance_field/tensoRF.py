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

import torch
from torch import nn
import torch.nn.functional as F

from util.misc import get_parameters_from_state_dict


class TensorVMSplit(nn.Module):

    def __init__(self, grid_dim, num_density_comps=(16, 16, 16), num_appearance_comps=(48, 48, 48), num_semantics_comps=None, dim_appearance=27,
                 dim_semantics=27, splus_density_shift=-10, pe_view=2, pe_feat=2, dim_mlp_color=128, dim_mlp_semantics=128, num_semantic_classes=0,
                 output_mlp_semantics=torch.nn.Softmax(dim=-1), dim_mlp_instance=256, dim_feature_instance=None, use_semantic_mlp=False, use_feature_reg=False):
        super().__init__()
        self.num_density_comps = num_density_comps
        self.num_appearance_comps = num_appearance_comps
        self.num_semantics_comps = num_semantics_comps
        self.dim_appearance = dim_appearance
        self.dim_semantics = dim_semantics
        self.dim_feature_instance = dim_feature_instance
        self.num_semantic_classes = num_semantic_classes
        self.splus_density_shift = splus_density_shift
        self.use_semantic_mlp = use_semantic_mlp
        self.use_feature_reg = use_feature_reg and use_semantic_mlp
        self.pe_view, self.pe_feat = pe_view, pe_feat
        self.dim_mlp_color = dim_mlp_color
        self.matrix_mode = [[0, 1], [0, 2], [1, 2]]
        self.vector_mode = [2, 1, 0]
        self.density_plane, self.density_line = self.init_one_svd(self.num_density_comps, grid_dim, 0.1)
        self.appearance_plane, self.appearance_line = self.init_one_svd(self.num_appearance_comps, grid_dim, 0.1)
        self.appearance_basis_mat = torch.nn.Linear(sum(self.num_appearance_comps), self.dim_appearance, bias=False)
        self.render_appearance_mlp = MLPRenderFeature(dim_appearance, 3, pe_view, pe_feat, dim_mlp_color)
        self.semantic_plane, self.semantic_line, self.semantic_basis_mat = None, None, None
        self.instance_plane, self.instance_line, self.instance_basis_mat = None, None, None
        if self.dim_feature_instance is not None:
            self.render_instance_mlp = MLPRenderInstanceFeature(3, dim_feature_instance, num_mlp_layers=4, dim_mlp=dim_mlp_instance, output_activation=torch.nn.Identity())
        if self.num_semantics_comps is not None and not use_semantic_mlp:
            self.semantic_plane, self.semantic_line = self.init_one_svd(self.num_semantics_comps, grid_dim, 0.1)
            self.semantic_basis_mat = torch.nn.Linear(sum(self.num_semantics_comps), self.dim_semantics, bias=False)
            self.render_semantic_mlp = MLPRenderFeature(self.dim_semantics, num_semantic_classes, 0, 0, dim_mlp_semantics, output_activation=output_mlp_semantics)
        elif use_semantic_mlp:
            self.render_semantic_mlp = (MLPRenderSemanticFeature if not self.use_feature_reg else MLPRenderSemanticFeatureWithRegularization)(3, num_semantic_classes, output_activation=output_mlp_semantics)

    def init_one_svd(self, n_components, grid_resolution, scale):
        plane_coef, line_coef = [], []
        for i in range(len(self.vector_mode)):
            vec_id = self.vector_mode[i]
            mat_id_0, mat_id_1 = self.matrix_mode[i]
            plane_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_components[i], grid_resolution[mat_id_1], grid_resolution[mat_id_0])), requires_grad=True))
            line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_components[i], grid_resolution[vec_id], 1)), requires_grad=True))
        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)

    def get_coordinate_plane_line(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matrix_mode[0]], xyz_sampled[..., self.matrix_mode[1]], xyz_sampled[..., self.matrix_mode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vector_mode[0]], xyz_sampled[..., self.vector_mode[1]], xyz_sampled[..., self.vector_mode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        return coordinate_plane, coordinate_line

    def compute_density_without_activation(self, xyz_sampled):
        coordinate_plane, coordinate_line = self.get_coordinate_plane_line(xyz_sampled)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature + self.splus_density_shift

    def compute_density(self, xyz_sampled):
        return F.softplus(self.compute_density_without_activation(xyz_sampled))

    def compute_feature(self, xyz_sampled, feature_plane, feature_line, basis_mat):
        coordinate_plane, coordinate_line = self.get_coordinate_plane_line(xyz_sampled)
        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(feature_plane)):
            plane_coef_point.append(F.grid_sample(feature_plane[idx_plane], coordinate_plane[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(feature_line[idx_plane], coordinate_line[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        return basis_mat((plane_coef_point * line_coef_point).T)

    def compute_appearance_feature(self, xyz_sampled):
        return self.compute_feature(xyz_sampled, self.appearance_plane, self.appearance_line, self.appearance_basis_mat)

    def compute_semantic_feature(self, xyz_sampled):
        if self.use_semantic_mlp:
            return xyz_sampled
        return self.compute_feature(xyz_sampled, self.semantic_plane, self.semantic_line, self.semantic_basis_mat)

    def render_instance_grid(self, xyz_sampled):
        retval = F.one_hot(F.grid_sample(self.instance_grid, xyz_sampled.unsqueeze(0).unsqueeze(0).unsqueeze(0), align_corners=True, padding_mode="border", mode='nearest').squeeze().long(), num_classes=self.dim_feature_instance).float()
        retval = torch.log(retval + 1e-8)
        return retval

    def compute_instance_feature(self, xyz_sampled):
        return self.render_instance_mlp(xyz_sampled)

    @torch.no_grad()
    def shrink(self, t_l, b_r):
        for i in range(len(self.vector_mode)):
            mode0 = self.vector_mode[i]
            self.density_line[i] = torch.nn.Parameter(self.density_line[i].data[..., t_l[mode0]:b_r[mode0], :])
            self.appearance_line[i] = torch.nn.Parameter(self.appearance_line[i].data[..., t_l[mode0]:b_r[mode0], :])
            if self.semantic_line is not None:
                self.semantic_line[i] = torch.nn.Parameter(self.semantic_line[i].data[..., t_l[mode0]:b_r[mode0], :])
            mode0, mode1 = self.matrix_mode[i]
            self.density_plane[i] = torch.nn.Parameter(self.density_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])
            self.appearance_plane[i] = torch.nn.Parameter(self.appearance_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])
            if self.semantic_plane is not None:
                self.semantic_plane[i] = torch.nn.Parameter(self.semantic_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.appearance_plane, self.appearance_line = self.upsample_plane_line(self.appearance_plane, self.appearance_line, res_target)
        self.density_plane, self.density_line = self.upsample_plane_line(self.density_plane, self.density_line, res_target)
        if self.semantic_plane is not None:
            self.semantic_plane, self.semantic_line = self.upsample_plane_line(self.semantic_plane, self.semantic_line, res_target)

    @torch.no_grad()
    def upsample_plane_line(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vector_mode)):
            vec_id = self.vector_mode[i]
            mat_id_0, mat_id_1 = self.matrix_mode[i]
            plane_coef[i] = torch.nn.Parameter(F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear', align_corners=True))
            line_coef[i] = torch.nn.Parameter(F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
        return plane_coef, line_coef

    def get_optimizable_parameters(self, lr_grid, lr_net, weight_decay=0):
        grad_vars = [{'params': self.density_line, 'lr': lr_grid, 'weight_decay': weight_decay}, {'params': self.appearance_line, 'lr': lr_grid},
                     {'params': self.density_plane, 'lr': lr_grid, 'weight_decay': weight_decay}, {'params': self.appearance_plane, 'lr': lr_grid},
                     {'params': self.appearance_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_appearance_mlp.parameters(), 'lr': lr_net}]
        if self.semantic_plane is not None:
            grad_vars.extend([
                {'params': self.semantic_plane, 'lr': lr_grid}, {'params': self.semantic_line, 'lr': lr_grid},
                {'params': self.semantic_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}])
        elif self.render_semantic_mlp is not None:
            grad_vars.extend([{'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}])
        return grad_vars

    def get_optimizable_segment_parameters(self, lr_grid, lr_net, _weight_decay=0):
        grad_vars = []
        if self.semantic_plane is not None:
            grad_vars.extend([
                {'params': self.semantic_plane, 'lr': lr_grid}, {'params': self.semantic_line, 'lr': lr_grid},
                {'params': self.semantic_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}])
        elif self.render_semantic_mlp is not None:
            grad_vars.extend([{'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}])
        return grad_vars

    def get_optimizable_instance_parameters(self, lr_grid, lr_net):
        return [
            {'params': self.render_instance_mlp.parameters(), 'lr': lr_net, 'weight_decay': 1e-3}
        ]

    def tv_loss_density(self, regularizer):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + regularizer(self.density_plane[idx]) * 1e-2  # + regularizer(self.density_line[idx]) * 1e-3
        return total

    def tv_loss_appearance(self, regularizer):
        total = 0
        for idx in range(len(self.appearance_plane)):
            total = total + regularizer(self.appearance_plane[idx]) * 1e-2  # + regularizer(self.appearance_line[idx]) * 1e-3
        return total

    def tv_loss_semantics(self, regularizer):
        total = 0
        if self.semantic_plane is not None:
            for idx in range(len(self.semantic_plane)):
                total = total + regularizer(self.semantic_plane[idx]) * 1e-2 + regularizer(self.semantic_line[idx]) * 1e-3
        return total

    def load_weights_debug(self, weights):
        self.density_plane.load_state_dict(get_parameters_from_state_dict(weights, 'density_plane'))
        self.density_line.load_state_dict(get_parameters_from_state_dict(weights, 'density_line'))
        self.appearance_plane.load_state_dict(get_parameters_from_state_dict(weights, 'appearance_plane'))
        self.appearance_line.load_state_dict(get_parameters_from_state_dict(weights, 'appearance_line'))
        self.appearance_basis_mat.load_state_dict(get_parameters_from_state_dict(weights, 'appearance_basis_mat'))
        self.render_appearance_mlp.load_state_dict(get_parameters_from_state_dict(weights, 'render_appearance_mlp'))
        if self.num_semantics_comps is not None:
            if self.semantic_plane is not None:
                self.semantic_plane.load_state_dict(get_parameters_from_state_dict(weights, 'semantic_plane'))
                self.semantic_line.load_state_dict(get_parameters_from_state_dict(weights, 'semantic_line'))
                self.semantic_basis_mat.load_state_dict(get_parameters_from_state_dict(weights, 'semantic_basis_mat'))
            self.render_semantic_mlp.load_state_dict(get_parameters_from_state_dict(weights, 'render_semantic_mlp'))
        if self.dim_feature_instance is not None:
            if self.instance_plane is not None:
                self.instance_plane.load_state_dict(get_parameters_from_state_dict(weights, 'instance_plane'))
                self.instance_line.load_state_dict(get_parameters_from_state_dict(weights, 'instance_line'))
                self.instance_basis_mat.load_state_dict(get_parameters_from_state_dict(weights, 'instance_basis_mat'))
            self.render_instance_mlp.load_state_dict(get_parameters_from_state_dict(weights, 'render_instance_mlp'))


class MLPRenderFeature(torch.nn.Module):

    def __init__(self, in_channels, out_channels=3, pe_view=2, pe_feat=2, dim_mlp_color=128, output_activation=torch.sigmoid):
        super().__init__()
        self.pe_view = pe_view
        self.pe_feat = pe_feat
        self.output_channels = out_channels
        self.view_independent = self.pe_view == 0 and self.pe_feat == 0
        self.in_feat_mlp = 2 * pe_view * 3 + 2 * pe_feat * in_channels + in_channels + (3 if not self.view_independent else 0)
        self.output_activation = output_activation
        layer1 = torch.nn.Linear(self.in_feat_mlp, dim_mlp_color)
        layer2 = torch.nn.Linear(dim_mlp_color, dim_mlp_color)
        layer3 = torch.nn.Linear(dim_mlp_color, out_channels)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, viewdirs, features):
        indata = [features]
        if not self.view_independent:
            indata.append(viewdirs)
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(features, self.pe_feat)]
        if self.pe_view > 0:
            indata += [MLPRenderFeature.positional_encoding(viewdirs, self.pe_view)]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        out = self.output_activation(out)
        return out

    @staticmethod
    def positional_encoding(positions, freqs):
        freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
        pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] + (freqs * positions.shape[-1],))
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts


class MLPRenderInstanceFeature(torch.nn.Module):

    def __init__(self, in_channels, out_channels, num_mlp_layers=5, dim_mlp=256, output_activation=torch.nn.Softmax(dim=-1)):
        super().__init__()
        self.output_channels = out_channels
        self.output_activation = output_activation
        layers = [torch.nn.Linear(in_channels, dim_mlp)]
        for i in range(num_mlp_layers - 2):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(dim_mlp, dim_mlp))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(dim_mlp, out_channels))
        self.mlp = torch.nn.Sequential(*layers)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, feat_xyz):
        out = self.mlp(feat_xyz)
        out = self.output_activation(out)
        return out


class MLPRenderSemanticFeature(torch.nn.Module):

    def __init__(self, in_channels, out_channels, pe_feat=0, num_mlp_layers=5, dim_mlp=256, output_activation=torch.nn.Identity()):
        super().__init__()
        self.output_channels = out_channels
        self.output_activation = output_activation
        self.pe_feat = pe_feat
        self.in_feat_mlp = 2 * pe_feat * in_channels + in_channels
        layers = [torch.nn.Linear(self.in_feat_mlp, dim_mlp)]
        for i in range(num_mlp_layers - 2):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(dim_mlp, dim_mlp))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(dim_mlp, out_channels))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, _dummy, feat_xyz):
        indata = [feat_xyz]
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(feat_xyz, self.pe_feat)]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        out = self.output_activation(out)
        return out


class MLPRenderSemanticFeatureWithRegularization(torch.nn.Module):

    def __init__(self, in_channels, out_channels, pe_feat=0, num_mlp_layers=5, dim_mlp=256, output_activation=torch.nn.Identity()):
        super().__init__()
        self.output_channels = out_channels
        self.output_activation = output_activation
        self.pe_feat = pe_feat
        self.in_feat_mlp = 2 * pe_feat * in_channels + in_channels
        layers = [torch.nn.Linear(self.in_feat_mlp, dim_mlp)]
        for i in range(num_mlp_layers - 3):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(dim_mlp, dim_mlp))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(dim_mlp, 384))
        self.mlp_backbone = torch.nn.Sequential(*layers)
        self.head_class = torch.nn.Linear(384, out_channels)

    def forward(self, _dummy, feat_xyz):
        out = self.get_backbone_feats(feat_xyz)
        out = self.head_class(out)
        out = self.output_activation(out)
        return out

    def get_backbone_feats(self, feat_xyz):
        indata = [feat_xyz]
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(feat_xyz, self.pe_feat)]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp_backbone(mlp_in)
        return out


def render_features_direct(_viewdirs, appearance_features):
    return appearance_features


def render_features_direct_with_softmax(_viewdirs, appearance_features):
    return torch.nn.Softmax(dim=-1)(appearance_features)
