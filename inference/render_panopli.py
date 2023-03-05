# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import sys
import random
import omegaconf
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
from dataset import PanopLiDataset, create_segmentation_data_panopli
from model.radiance_field.tensoRF import TensorVMSplit
from model.renderer.panopli_tensoRF_renderer import TensoRFRenderer
from trainer import visualize_panoptic_outputs
from util.camera import distance_to_depth
from util.misc import get_parameters_from_state_dict


def render_panopli_checkpoint(config, trajectory_name, test_only=False):
    output_dir = (Path("runs") / f"{Path(config.dataset_root).stem}_{trajectory_name if not test_only else 'test'}_{Path(config.experiment)}")
    print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    test_set = PanopLiDataset(Path(config.dataset_root), "test", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir='m2f_semantics', instance_dir='m2f_instance',
                              instance_to_semantic_key='m2f_instance_to_semantic', create_seg_data_func=create_segmentation_data_panopli, subsample_frames=config.subsample_frames)
    H, W, alpha = config.image_dim[0], config.image_dim[1], 0.65
    # whether to render the test set or a predefined trajectory through the scene
    if test_only:
        trajectory_set = test_set
    else:
        trajectory_set = test_set.get_trajectory_set(trajectory_name, True)
    trajectory_loader = DataLoader(trajectory_set, shuffle=False, num_workers=0, batch_size=1)
    checkpoint = torch.load(config.resume, map_location="cpu")
    state_dict = checkpoint['state_dict']
    total_classes = len(test_set.segmentation_data.bg_classes) + len(test_set.segmentation_data.fg_classes)
    output_mlp_semantics = torch.nn.Identity() if config.semantic_weight_mode != "softmax" else torch.nn.Softmax(dim=-1)
    model = TensorVMSplit([config.min_grid_dim, config.min_grid_dim, config.min_grid_dim], num_semantics_comps=(32, 32, 32),
                           num_semantic_classes=total_classes, dim_feature_instance=config.max_instances,
                           output_mlp_semantics=output_mlp_semantics, use_semantic_mlp=config.use_mlp_for_semantics)
    renderer = TensoRFRenderer(test_set.scene_bounds, [config.min_grid_dim, config.min_grid_dim, config.min_grid_dim], semantic_weight_mode=config.semantic_weight_mode)
    renderer.load_state_dict(get_parameters_from_state_dict(state_dict, "renderer"))
    for epoch in config.grid_upscale_epochs[::-1]:
        if checkpoint['epoch'] >= epoch:
            model.upsample_volume_grid(renderer.grid_dim)
            renderer.update_step_size(renderer.grid_dim)
            break
    model.load_state_dict(get_parameters_from_state_dict(state_dict, "model"))
    model = model.to(device)
    renderer = renderer.to(device)

    # disable this for fast rendering (just add more steps along the ray)
    renderer.update_step_ratio(renderer.step_ratio * 0.5)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(trajectory_loader)):
            batch['rays'] = batch['rays'].squeeze(0).to(device)
            concated_outputs = []
            outputs = []
            # infer semantics and surrogate ids
            for i in range(0, batch['rays'].shape[0], config.chunk):
                out_rgb_, out_semantics_, out_instances_, out_depth_, _, _ = renderer(model, batch['rays'][i: i + config.chunk], config.perturb, test_set.white_bg, False)
                outputs.append([out_rgb_, out_semantics_, out_instances_, out_depth_])
            for i in range(len(outputs[0])):
                concated_outputs.append(torch.cat([outputs[j][i] for j in range(len(outputs))], dim=0))
            p_rgb, p_semantics, p_instances, p_dist = concated_outputs
            p_depth = distance_to_depth(test_set.intrinsics[0], p_dist.view(H, W))
            # create surrogate ids
            p_instances = create_instances_from_semantics(p_instances, p_semantics, test_set.segmentation_data.fg_classes)
            (output_dir / "vis_semantics_and_surrogate").mkdir(exist_ok=True)
            (output_dir / "pred_semantics").mkdir(exist_ok=True)
            (output_dir / "pred_surrogateid").mkdir(exist_ok=True)

            stack = visualize_panoptic_outputs(p_rgb, p_semantics, p_instances, p_depth, None, None, None, H, W, thing_classes=test_set.segmentation_data.fg_classes, visualize_entropy=False)
            output_semantics_with_invalid = p_semantics.detach().argmax(dim=1)
            grid = (make_grid(stack, value_range=(0, 1), normalize=True, nrow=5).permute((1, 2, 0)).contiguous() * 255).cpu().numpy().astype(np.uint8)

            name = f"{test_set.all_frame_names[test_set.val_indices[batch_idx]]}.png" if test_only else f"{batch_idx:04d}.png"
            Image.fromarray(grid).save(output_dir / "vis_semantics_and_surrogate" / name)
            Image.fromarray(output_semantics_with_invalid.reshape(H, W).cpu().numpy().astype(np.uint8)).save(output_dir / "pred_semantics" / name)
            Image.fromarray(p_instances.argmax(dim=1).reshape(H, W).cpu().numpy().astype(np.uint16)).save(output_dir / "pred_surrogateid" / name)


def create_instances_from_semantics(instances, semantics, thing_classes):
    stuff_mask = ~torch.isin(semantics.argmax(dim=1), torch.tensor(thing_classes).to(semantics.device))
    padded_instances = torch.ones((instances.shape[0], instances.shape[1] + 1), device=instances.device) * -float('inf')
    padded_instances[:, 1:] = instances
    padded_instances[stuff_mask, 0] = float('inf')
    return padded_instances


if __name__ == "__main__":
    # needs a predefined trajectory named trajectory_blender in case test_only = False
    _trajectory_name = "trajectory_blender"
    cfg = omegaconf.OmegaConf.load(Path(sys.argv[1]).parents[1] / "config.yaml")
    cfg.resume = sys.argv[1]
    test_mode = False if len(sys.argv) == 2 else sys.argv[2] == "True"
    cfg.image_dim = [256, 384]
    if isinstance(cfg.image_dim, int):
        cfg.image_dim = [cfg.image_dim, cfg.image_dim]
    if test_mode:
        render_panopli_checkpoint(cfg, _trajectory_name, test_only=True)
    else:
        render_panopli_checkpoint(cfg, _trajectory_name, test_only=False)
