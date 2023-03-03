# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import random
import pickle
import numpy as np
import trimesh
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from util.camera import compute_world2normscene, unproject_2d_3d
from util.distinct_colors import DistinctColors
from util.misc import visualize_points, EasyDict, create_box, visualize_cameras, visualize_points_as_pts
from util.ray import get_ray_directions_with_intrinsics, get_rays, rays_intersect_sphere


def create_segmentation_data_base(num_semantic_classes, instance_to_semantics, num_instances):
    fg_classes = sorted(list(set(list(instance_to_semantics.values())) - {0}))
    bg_classes = sorted(list(set(list(range(num_semantic_classes))) - set(fg_classes)))
    return EasyDict({
        'fg_classes': fg_classes,
        'bg_classes': bg_classes,
        'instance_to_semantics': instance_to_semantics,
        'num_semantic_classes': num_semantic_classes,
        'num_instances': num_instances
    })


def create_segmentation_data_sem(num_semantic_classes, instance_to_semantics, num_instances):
    seg_data = create_segmentation_data_base(num_semantic_classes, instance_to_semantics, num_instances)
    seg_data.num_instances = len(seg_data.fg_classes)
    seg_data.instance_to_semantics = {(i + 1): k for i, k in enumerate(seg_data.fg_classes)}
    seg_data.instance_to_semantics[0] = 0
    return seg_data


class BaseDataset(Dataset):

    def __init__(self, root_dir, split, image_dim, max_depth, overfit=False, num_val_samples=8, load_depth=False, load_feat=False, instance_dir='filtered_pano_instance', semantics_dir='filtered_pano_sem',
                 instance_to_semantic_key='instance_to_semantic', create_seg_data_func=create_segmentation_data_base, subsample_frames=1, run_setup_data=True):
        self.root_dir = root_dir
        self.split = split
        self.subsample_frames = subsample_frames
        self.image_dim = image_dim
        self.transform = T.ToTensor()
        self.max_depth = max_depth
        self.white_bg = False
        self.load_depth = load_depth
        self.load_feat = load_feat
        self.instance_directory = instance_dir
        self.semantics_directory = semantics_dir
        self.instance_to_semantic_key = instance_to_semantic_key
        self.create_segmentation_data = create_seg_data_func
        self.train_indices = self.val_indices = []
        self.num_val_samples = num_val_samples
        self.overfit = overfit
        self.cam2scenes = {}
        self.world2scene = None
        self.scene2normscene = None
        self.intrinsics = {}
        self.cam2normscene = {}
        self.normscene_scale = None
        self.all_rays = []
        self.all_rgbs = []
        self.all_depths = []
        self.all_semantics = []
        self.all_instances = []
        self.all_masks = []
        self.scene_bounds = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
        self.bounding_boxes = None
        self.segmentation_data = None
        self.invalid_class = None
        if run_setup_data:
            self.setup_data()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays) // torch.cuda.device_count() * torch.cuda.device_count()
        if self.split == 'val' or self.split == 'test':
            return len(self.val_indices) // torch.cuda.device_count() * torch.cuda.device_count()
        raise NotImplementedError

    def __getitem__(self, idx):
        sample = None
        if self.split == "train":
            sample = {
                "rays": self.all_rays[idx, :8],
                "rgbs": self.all_rgbs[idx],
                "semantics": self.all_semantics[idx],
                "instances": self.all_instances[idx],
                "probabilities": self.all_probabilities[idx],
                "confidences": self.all_confidences[idx],
                "feats": self.all_feats[idx] if self.load_feat else torch.zeros(1, 1),
                "mask": self.all_masks[idx],
            }
        elif self.split == "val" or self.split == "test":
            sample_idx = self.val_indices[idx % len(self.val_indices)]
            image, rays, semantics, instances, depth, _, prob, conf, _, mask = self.load_sample(sample_idx)
            sample = {
                "rays": rays,
                "rgbs": image,
                "semantics": semantics,
                "instances": instances,
                "probabilities": prob,
                "confidences": conf,
                "mask": mask,
            }
        return sample

    def setup_data(self):
        scene_annotation = pickle.load(open(self.root_dir / 'scene_annotation.pkl', 'rb'))
        sample_indices = scene_annotation['sample_inds']

        if self.overfit:
            self.train_indices = self.val_indices = [sample_indices[0], sample_indices[1], sample_indices[2], sample_indices[3]]
        else:
            if "val_inds" in scene_annotation:
                self.val_indices = scene_annotation["val_inds"]
            else:
                self.val_indices = np.random.choice(sample_indices, min(len(sample_indices), self.num_val_samples))
            self.train_indices = [sample_index for sample_index in sample_indices if sample_index not in self.val_indices]

        self.world2scene = scene_annotation['world2scene']
        dims, intrinsics, cam2scene = [], [], []

        for sample_index in sample_indices:
            sample_annotation = pickle.load(open(self.root_dir / 'annotation' / f'{sample_index}.pkl', 'rb'))
            intrinsic, img_h, img_w = sample_annotation['intrinsics'], sample_annotation['height'], sample_annotation['width']
            cam2world = sample_annotation['cam2world']
            cam2scene.append(torch.from_numpy(self.world2scene @ cam2world).float())
            self.cam2scenes[sample_index] = cam2scene[-1]
            dims.append([img_h, img_w])
            intrinsics.append(torch.from_numpy(intrinsic))
            self.intrinsics[sample_index] = intrinsic
            self.intrinsics[sample_index] = torch.from_numpy(np.diag([self.image_dim[1] / img_w, self.image_dim[0] / img_h, 1]) @ self.intrinsics[sample_index]).float()

        self.scene2normscene = compute_world2normscene(
            torch.Tensor(dims).float(),
            torch.stack(intrinsics).float(),
            torch.stack(cam2scene).float(),
            max_depth=self.max_depth,
            rescale_factor=1.0
        )

        self.normscene_scale = self.scene2normscene[0, 0]
        for sample_index in sample_indices:
            self.cam2normscene[sample_index] = self.scene2normscene @ self.cam2scenes[sample_index]

        pkl_segmentation_data = pickle.load(open(self.root_dir / 'filtered_instances.pkl', 'rb'))
        self.bounding_boxes = process_bounding_box_dict(pkl_segmentation_data['bboxes'], (self.scene2normscene @ self.world2scene).numpy())
        num_instances = self.bounding_boxes.ids.shape[0]
        self.segmentation_data = self.create_segmentation_data(pkl_segmentation_data['num_semantic_classes'], pkl_segmentation_data[self.instance_to_semantic_key], num_instances)
        if self.split == "train":
            for sample_index in self.train_indices:
                image, rays, semantics, instances, depth, _, _, _, _, _ = self.load_sample(sample_index)
                self.all_rgbs.append(image)
                self.all_rays.append(rays)
                self.all_semantics.append(semantics)
                self.all_instances.append(instances)
                if self.load_depth:
                    self.all_depths.append(depth)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_semantics = torch.cat(self.all_semantics, 0)
            self.all_instances = torch.cat(self.all_instances, 0)
            if self.load_depth:
                self.all_depths = torch.cat(self.all_depths, 0)

            # to debug scene categories
            # print(hypersim_semantic_categories(torch.unique(self.all_semantics).numpy().tolist()))

    def load_sample(self, sample_index):
        cam2normscene = self.cam2normscene[sample_index]
        image = Image.open(self.root_dir / "rgb" / f"{sample_index}.jpg")
        # noinspection PyTypeChecker
        image = torch.from_numpy(np.array(image.resize(self.image_dim[::-1], Image.LANCZOS)) / 255).float()
        semantics = Image.open(self.root_dir / self.semantics_directory / f"{sample_index}.png")
        instances = Image.open(self.root_dir / self.instance_directory / f"{sample_index:04d}.png")
        # noinspection PyTypeChecker
        semantics = torch.from_numpy(np.array(semantics.resize(self.image_dim[::-1], Image.NEAREST))).long()
        # noinspection PyTypeChecker
        instances = torch.from_numpy(np.array(instances.resize(self.image_dim[::-1], Image.NEAREST))).long()
        # noinspection PyTypeChecker
        raw_depth = np.load(open(self.root_dir / 'depth' / f'{sample_index}.npy', 'rb'))
        raw_depth[raw_depth > (self.max_depth / self.normscene_scale.item())] = (self.max_depth / self.normscene_scale.item())
        # noinspection PyTypeChecker
        depth_cam = np.array(Image.fromarray(raw_depth).resize(self.image_dim[::-1], Image.NEAREST))
        depth_cam_s = self.normscene_scale * depth_cam
        depth = depth_cam_s.float()
        directions = get_ray_directions_with_intrinsics(self.image_dim[0], self.image_dim[1], self.intrinsics[sample_index].numpy())
        rays_o, rays_d = get_rays(directions, cam2normscene)

        sphere_intersection_displacement = rays_intersect_sphere(rays_o, rays_d, r=1)  # fg is in unit sphere

        rays = torch.cat(
            [rays_o, rays_d, 0.01 *
             torch.ones_like(rays_o[:, :1]), sphere_intersection_displacement[:, None], ], 1,
        )

        return image.reshape(-1, 3), rays, semantics.reshape(-1), instances.reshape(-1), depth.reshape(-1), \
               torch.from_numpy(depth_cam).reshape(-1), torch.ones(rays.shape[0]).bool(), torch.ones(rays.shape[0]).bool(), \
               torch.ones(rays.shape[0]).bool(), torch.ones(rays.shape[0]).bool()

    def export_point_cloud(self, output_path, subsample=1, export_semantics=False, export_bbox=True):
        all_rgbs = []
        all_world = []
        all_world_unscaled = []
        all_cameras_unscaled = []
        all_cameras = []
        if export_bbox:
            print('visualizing bounding boxes')
            color_manager_bbox = DistinctColors()
            all_bboxes = []
            # uncomment the next line to test noise in bboxes
            # bounding_boxes = BoundingBoxes.add_noise(self.bounding_boxes, 0.025, 0.25, 0.15)
            bounding_boxes = self.bounding_boxes
            for idx in range(self.bounding_boxes.ids.shape[0]):
                color = color_manager_bbox.get_color_fast_numpy(idx)
                bbox = create_box(bounding_boxes.positions[idx].numpy(),
                                  bounding_boxes.extents[idx].numpy(),
                                  bounding_boxes.orientations[idx].numpy(),
                                  color)
                all_bboxes.append(bbox)
            combined = trimesh.util.concatenate(all_bboxes)
            combined.export(output_path / "bboxes.obj")
        if export_semantics:
            all_semantics = []
            all_instances = []
            color_manager = DistinctColors()
            color_manager_instance = DistinctColors()
        for array_idx, sample_index in enumerate(tqdm(self.train_indices)):
            image, _, semantics, instances, _, depth, _, _, _, room_mask = self.load_sample(sample_index)
            world_points = unproject_2d_3d(self.cam2normscene[sample_index], self.intrinsics[sample_index], depth, self.image_dim)[room_mask, :]
            world_points_unscaled = unproject_2d_3d(self.cam2scenes[sample_index], self.intrinsics[sample_index], depth, self.image_dim)[room_mask, :]
            all_rgbs.append(image[room_mask, :])
            all_world.append(world_points)
            all_world_unscaled.append(world_points_unscaled)
            all_cameras_unscaled.append(self.cam2scenes[sample_index].unsqueeze(0))
            all_cameras.append(self.cam2normscene[sample_index].unsqueeze(0))
            if export_semantics:
                all_semantics.append(color_manager.apply_colors_fast_torch(semantics[room_mask]))
                all_instances.append(color_manager_instance.apply_colors_fast_torch(instances[room_mask]))
        all_world = torch.cat(all_world, 0)
        subsampled_indices = random.sample(list(range(all_world.shape[0])), int(all_world.shape[0] * subsample))
        all_world = all_world[subsampled_indices, :]
        all_rgbs = torch.cat(all_rgbs, 0)[subsampled_indices, :]
        all_world_unscaled = torch.cat(all_world_unscaled, 0)[subsampled_indices, :]
        all_cameras_unscaled = torch.cat(all_cameras_unscaled, 0)
        all_cameras = torch.cat(all_cameras, 0)
        print('visualizing rgb')
        visualize_points(all_world_unscaled, output_path / "pc_rgb.obj", all_rgbs)
        visualize_points_as_pts(all_world_unscaled, output_path / "pc_rgb.pts", (all_rgbs * 255).int())
        print('visualizing rgb scaled')
        visualize_points(all_world, output_path / "pc_rgb_scaled.obj", all_rgbs)
        visualize_points_as_pts(all_world, output_path / "pc_rgb_scaled.pts", (all_rgbs * 255).int())
        if export_semantics:
            print('visualizing semantics')
            all_semantics = torch.cat(all_semantics, 0)[subsampled_indices, :]
            all_instances = torch.cat(all_instances, 0)[subsampled_indices, :]
            visualize_points(all_world, output_path / "pc_sem.obj", all_semantics)
            visualize_points(all_world, output_path / "pc_instance.obj", all_instances)
        print('visualizing cameras')
        visualize_cameras(output_path / "pc_cam_scaled.obj", all_cameras)
        visualize_cameras(output_path / "pc_cam.obj", all_cameras_unscaled)

    def get_trajectory_set(self, trajectory_name, norm_scene=False, hotfix=torch.eye(4)):
        if norm_scene:
            return MainerTrajectoryDataset(self, trajectory_name, self.root_dir / "trajectories", self.scene2normscene, hotfix)
        else:
            return MainerTrajectoryDataset(self, trajectory_name, self.root_dir / "trajectories", hotfix=hotfix)

    def get_canonical_set(self, trajectory_name, cameras):
        return MainerTrajectoryDataset(self, trajectory_name, cameras)

    @property
    def num_instances(self):
        return self.segmentation_data.num_instances


class MainerTrajectoryDataset(Dataset):

    def __init__(self, mainer_base, trajectory_name, resource, scene2normscene=torch.eye(4), hotfix=torch.eye(4)):
        self.base = mainer_base
        self.trajectory_name = trajectory_name
        self.scene2normscene = scene2normscene
        if isinstance(resource, list):
            self.trajectories = resource
        else:
            with open(resource / f"{trajectory_name}.pkl", "rb") as fptr:
                self.trajectories = pickle.load(fptr)
        self.hotfix = hotfix

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        cam2normscene = self.scene2normscene @ self.hotfix @ torch.from_numpy(self.trajectories[idx]).float()
        directions = get_ray_directions_with_intrinsics(self.base.image_dim[0], self.base.image_dim[1], self.base.intrinsics[0].numpy())
        rays_o, rays_d = get_rays(directions, cam2normscene)

        sphere_intersection_displacement = rays_intersect_sphere(rays_o, rays_d, r=1)

        rays = torch.cat(
            [rays_o, rays_d, 0.01 *
             torch.ones_like(rays_o[:, :1]), sphere_intersection_displacement[:, None], ], 1,
        )

        return {
            "name": f"{idx:04d}",
            "rays": rays,
        }


class InconsistentBaseDataset(BaseDataset):

    def __init__(self, root_dir, split, image_dim, max_depth, semantic_class, overfit=False, num_val_samples=8, max_rays=512):
        super().__init__(root_dir, split, image_dim, max_depth, overfit, num_val_samples, instance_dir='filtered_pano_instance_inc', instance_to_semantic_key='instance_to_semantic_inc',
                         create_seg_data_func=create_segmentation_data_base)
        print('Preparing InconsistentMainerDataset...')
        all_semantics_view = self.all_semantics.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        all_rays_view = self.all_rays.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1], -1)
        all_instances_view = self.all_instances.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        all_rays = []
        all_instances = []
        for i in range(len(self.train_indices)):
            mask = all_semantics_view[i] == semantic_class
            if mask.sum() > 0:
                all_rays.append(all_rays_view[i][mask, :])
                all_instances.append(all_instances_view[i][mask])

        self.all_rays = all_rays
        self.all_instances = all_instances
        self.max_rays = max_rays

    def __getitem__(self, idx):
        selected_rays = self.all_rays[idx]
        selected_instances = self.all_instances[idx]
        if selected_rays.shape[0] > self.max_rays:
            sampled_indices = random.sample(range(selected_rays.shape[0]), self.max_rays)
            selected_rays = selected_rays[sampled_indices, :]
            selected_instances = selected_instances[sampled_indices]
        sample = {
            f"rays": selected_rays,
            f"instances": selected_instances,
        }
        return sample

    def __len__(self):
        return len(self.all_rays)

    @staticmethod
    def collate_fn(batch):
        return {
            "rays": [x["rays"] for x in batch],
            "instances": [x["instances"] for x in batch],
        }


class InconsistentSingleBaseDataset(BaseDataset):

    def __init__(self, root_dir, split, image_dim, max_depth, overfit=False, num_val_samples=8, max_rays=512):
        super().__init__(root_dir, split, image_dim, max_depth, overfit, num_val_samples, instance_dir='filtered_pano_instance_inc', instance_to_semantic_key='instance_to_semantic_inc',
                         create_seg_data_func=create_segmentation_data_base)
        print('Preparing InconsistentMainerDataset...')
        all_rays_view = self.all_rays.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1], -1)
        all_instances_view = self.all_instances.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        all_rays = []
        all_instances = []
        for i in range(len(self.train_indices)):
            mask = all_instances_view[i] != 0
            if mask.sum() > 0:
                all_rays.append(all_rays_view[i][mask, :])
                all_instances.append(all_instances_view[i][mask])

        self.all_rays = all_rays
        self.all_instances = all_instances
        self.max_rays = max_rays

    def __getitem__(self, idx):
        selected_rays = self.all_rays[idx]
        selected_instances = self.all_instances[idx]
        if selected_rays.shape[0] > self.max_rays:
            sampled_indices = random.sample(range(selected_rays.shape[0]), self.max_rays)
            selected_rays = selected_rays[sampled_indices, :]
            selected_instances = selected_instances[sampled_indices]
        sample = {
            f"rays": selected_rays,
            f"instances": selected_instances,
        }
        return sample

    def __len__(self):
        return len(self.all_rays)

    @staticmethod
    def collate_fn(batch):
        return {
            "rays": [x["rays"] for x in batch],
            "instances": [x["instances"] for x in batch],
        }


def process_bounding_box_dict(bbdict, world2scene):
    valid_keys = [k for k in bbdict.keys() if bbdict[k]['position'][0] != float('inf')]
    num_bbs = len(valid_keys)
    bb_ids = torch.zeros(num_bbs).long()
    bb_extents = torch.zeros((num_bbs, 3)).float()
    bb_positions = torch.zeros((num_bbs, 3)).float()
    bb_orientations = torch.zeros((num_bbs, 3, 3)).float()
    scale = np.linalg.norm(world2scene[:3, 0])
    rotation = world2scene[:3, :3] / scale
    for idx, key in enumerate(valid_keys):
        bb_ids[idx] = key
        bb_extents[idx] = torch.from_numpy(bbdict[key]['extent']).float()
        bb_positions[idx] = torch.from_numpy(bbdict[key]['position']).float()
        bb_orientations[idx] = torch.from_numpy(rotation @ bbdict[key]['orientation']).float()
    return EasyDict({
        'ids': bb_ids,
        'orientations': bb_orientations,
        'positions': (torch.from_numpy(world2scene[:3, :3]).float() @ bb_positions.T + torch.from_numpy(world2scene[:3, 3:4]).float()).T,
        'extents': scale * bb_extents * 1.05,
    })
