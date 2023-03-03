# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import json
import math
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from scipy.stats import mode

import numpy as np
import plyfile
import torch
import trimesh
from PIL import Image
from tqdm import tqdm
from transforms3d.axangles import axangle2mat

from dataset.preprocessing.preprocess_scannet import get_thing_semantics, get_classnames, visualize_mask_folder, \
    create_validation_set, map_panoptic_coco, mmdet_export_fixes
from util.distinct_colors import DistinctColors
from util.misc import create_box
from util.transforms import hmg

from_dm_nerf_path = Path("data/replica/raw/from_dm_nerf")
from_semantic_nerf_path = Path("data/replica/raw/from_semantic_nerf")
from_replica_github = Path("data/replica/raw/from_github/replica_v1")


scene_specific_fixes_rotation = {
    "room_2": [math.radians(9.08), -0.132, -0.991, -0.010]
}
# fix ambiguous / incorrect annotations
scene_specific_fixes_objectid = {
    "room_0": {},
    "room_1": {},
    "room_2": {},
    "office_0": {
        3: 15,
        30: 15,
    },
    "office_2": {
        69: 15,
        2: 15,
        0: 8,
        3: 8
    },
    "office_3": {
        55: 6,
        8: 6,
        12: 21,
        88: 15,
        89: 15,
        111: 12,
        103: 12,
        39: 12,
        97: 12,
        0: 8,
    },
    "office_4": {
        10: 7,
        51: 6,
        52: 6,
        16: 15,
        18: 15,
        14: 5,
    }
}


def copy_images_to_src_folder(src_folder):
    Path(src_folder).mkdir(exist_ok=True)
    list_train = sorted(list(Path(from_semantic_nerf_path / src_folder.stem / "Sequence_1", "rgb").iterdir()), key=lambda x: int(x.stem.split('_')[1]))
    Path(src_folder, "color").mkdir(exist_ok=True)
    Path(src_folder, "depth").mkdir(exist_ok=True)
    Path(src_folder, "panoptic").mkdir(exist_ok=True)
    for i in tqdm(range(len(list_train))):
        Image.open(list_train[i]).save(Path(src_folder, "color", f'0_{i:04d}.jpg'))
        shutil.copyfile(Path(from_semantic_nerf_path / src_folder.stem / "Sequence_1", "depth", f'depth_{i}.png'), Path(src_folder, "depth", f'0_{i:04d}.png'))


def copy_cam_to_src_folder(src_folder):
    list_all = sorted((src_folder / "color").iterdir(), key=lambda x: int(x.stem.split('_')[1]))
    W, H = Image.open(list_all[0]).size
    focal = W / 2.0
    Path(src_folder, "intrinsic").mkdir(exist_ok=True)
    Path(src_folder, "intrinsic", "intrinsic_color.txt").write_text(f"""{focal} 0.00 {(W - 1) * 0.5} 0.00\n0.00 {focal} {(H - 1) * 0.5} 0.00\n0.00 0.00 1.00 0.00\n0.00 0.00 0.00 1.00""")
    Path(src_folder, "pose").mkdir(exist_ok=True)
    rotation_fix = np.eye(4)
    if src_folder.stem in scene_specific_fixes_rotation:
        axangle = scene_specific_fixes_rotation[src_folder.stem]
        rotation_fix[:3, :3] = axangle2mat(axangle[1:4], axangle[0])

    tmesh = trimesh.load(from_replica_github / src_folder.stem / "mesh.ply")
    tmesh = tmesh.apply_transform(hmg(rotation_fix))
    translation_fix = hmg(np.eye(3))
    translation_fix[0, 3] = -(tmesh.bounds[0][0] + tmesh.bounds[1][0]) * 0.5
    translation_fix[1, 3] = -(tmesh.bounds[0][1] + tmesh.bounds[1][1]) * 0.5
    translation_fix[2, 3] = -tmesh.bounds[0][2]
    # tmesh = tmesh.apply_transform(translation_fix)
    # tmesh.export("test.obj")

    for i, line in enumerate(tqdm(Path(from_semantic_nerf_path, src_folder.stem, f"Sequence_{1}", "traj_w_c.txt").read_text().splitlines())):
        RT = np.array([float(x) for x in line.split()]).reshape(4, 4)
        RT = translation_fix @ rotation_fix @ RT
        Path(src_folder, "pose", f'0_{i:04d}.txt').write_text(f"""{RT[0, 0]} {RT[0, 1]} {RT[0, 2]} {RT[0, 3]}\n{RT[1, 0]} {RT[1, 1]} {RT[1, 2]} {RT[1, 3]}\n{RT[2, 0]} {RT[2, 1]} {RT[2, 2]} {RT[2, 3]}\n0.00 0.00 0.00 1.00""")


def print_used_labels():
    all_objects = []
    for room in from_dm_nerf_path.iterdir():
        sem_info = json.loads(Path(from_semantic_nerf_path / "semantic_info" / room.stem / "info_semantic.json").read_text())
        for objects in sem_info["objects"]:
            all_objects.append(objects['class_name'])
    all_objects = sorted(list(set(all_objects)))
    outfile = Path(f"resources/replica_to_scannet_reduced.csv")
    if not outfile.exists():
        outfile.write_text('\n'.join(all_objects))


def convert_from_semantics_and_instances_to_reduced(segments, object_id_to_scannet_label, is_thing, instance_to_semantic):
    unique_segment_ids = segments.unique()
    semantics = object_id_to_scannet_label[segments]
    instances = torch.zeros_like(semantics)
    for s in unique_segment_ids:
        if is_thing[object_id_to_scannet_label[s]]:
            instances[segments == s] = s
            instance_to_semantic[s] = object_id_to_scannet_label[s]
    return semantics, instances, instance_to_semantic


def get_replica_to_scannet(src_folder, scannet_classnames):
    replica_to_scannet = torch.zeros(300).long()
    replica_id = defaultdict(list)
    replica_sem_info = json.loads(Path(from_semantic_nerf_path / "semantic_info" / src_folder.stem / "info_semantic.json").read_text())
    for objects in replica_sem_info["objects"]:
        replica_id[objects['class_name']].append(objects['id'])
    pprint(replica_id)
    for cidx, cllist in enumerate([x.strip().split(',') for x in Path("resources/replica_to_scannet_reduced.csv").read_text().strip().splitlines()]):
        if cllist[0] in replica_id:
            for ob_id in replica_id[cllist[0]]:
                replica_to_scannet[ob_id] = scannet_classnames.index(cllist[1])
    for ob_id in scene_specific_fixes_objectid[src_folder.stem]:
        replica_to_scannet[ob_id] = scene_specific_fixes_objectid[src_folder.stem][ob_id]
    return replica_to_scannet


def map_gt_to_scannet(src_folder):
    list_train = sorted(list((src_folder / "color").iterdir()))
    thing_semantics = get_thing_semantics()
    sc_classname = get_classnames()
    replica_to_scannet = get_replica_to_scannet(src_folder, sc_classname)
    print(replica_to_scannet)
    instance_to_semantic = {}
    (src_folder / "rs_semantics").mkdir(exist_ok=True)
    (src_folder / "rs_instance").mkdir(exist_ok=True)
    for i, x in enumerate(tqdm(list_train)):
        # semantics = torch.from_numpy(np.array(Image.open(src_folder / "semantics" / f"{x.stem}.png")))
        instance = torch.from_numpy(np.array(Image.open(src_folder / "instance" / f"{x.stem}.png"))).long()
        semantics, instance, instance_to_semantic = convert_from_semantics_and_instances_to_reduced(instance, replica_to_scannet, thing_semantics, instance_to_semantic)
        Image.fromarray(semantics.numpy().astype(np.int8)).save(src_folder / "rs_semantics" / f"{x.stem}.png")
        Image.fromarray(instance.numpy().astype(np.int8)).save(src_folder / "rs_instance" / f"{x.stem}.png")
    export_dict = {}
    if (src_folder / "segmentation_data.pkl").exists():
        export_dict = pickle.load(open(src_folder / 'segmentation_data.pkl', 'rb'))
    export_dict['num_semantic_classes'] = len(thing_semantics)
    export_dict['instance_to_semantic'] = instance_to_semantic
    export_dict['fg_classes'] = [i for i, is_thing in enumerate(thing_semantics) if is_thing]
    export_dict['bg_classes'] = [i for i, is_thing in enumerate(thing_semantics) if not is_thing]

    pickle.dump(export_dict, open(src_folder / 'segmentation_data.pkl', 'wb'))


def extract_semantics_and_instances(src_folder):
    list_train = sorted(list(Path(from_semantic_nerf_path / src_folder.stem / "Sequence_1", "rgb").iterdir()), key=lambda x: int(x.stem.split('_')[1]))
    Path(src_folder, "semantics").mkdir(exist_ok=True)
    Path(src_folder, "instance").mkdir(exist_ok=True)
    for i in tqdm(range(len(list_train))):
        shutil.copyfile(from_dm_nerf_path / src_folder.stem / "semantic_class" / f"semantic_class_{i}.png", Path(src_folder, "semantics", f'0_{i:04d}.png'))
        shutil.copyfile(from_dm_nerf_path / src_folder.stem / "semantic_instance" / f"semantic_instance_{i}.png", Path(src_folder, "instance", f'0_{i:04d}.png'))


def mmdet_create_posed_images(src_folder, mmdet_folder, num_images=100):
    rotation_fix = np.eye(4)
    if src_folder.stem in mmdet_export_fixes:
        axangle = mmdet_export_fixes[src_folder.stem]["rotation"]
        if axangle is not None:
            rotation_fix[:3, :3] = axangle2mat(axangle[1:4], axangle[0])
        translation_fix = hmg(np.eye(3))
        translation_fix[:3, 3] = np.array(mmdet_export_fixes[src_folder.stem]["translation"])
        scale_fix = hmg(np.eye(3) * mmdet_export_fixes[src_folder.stem]["scale"])
    list_all = sorted((src_folder / "color").iterdir(), key=lambda x: int(x.stem.split('_')[1]) if "_" in x.stem else int(x.stem))
    if len(list_all) > num_images:
        selected = [i for i in range(0, len(list_all), int(len(list_all) / num_images))][:num_images]
    else:
        selected = list(range(len(list_all)))
    out_folder = mmdet_folder / "data" / "scannet" / "posed_images" / src_folder.stem
    out_folder.mkdir(exist_ok=True)
    for idx, i in enumerate(selected):
        image = Image.open(src_folder / "color" / f"{list_all[i].stem}.jpg")
        img_w, img_h = image.size
        image.resize((640, int(640 * image.size[1] / image.size[0])), Image.BICUBIC).save(out_folder / f"{idx:05d}.jpg")
        RT = np.array([[float(y.strip()) for y in x.strip().split()] for x in Path(src_folder / "pose" / f"{list_all[i].stem}.txt").read_text().splitlines() if x != ''])
        RT = translation_fix @ scale_fix @ rotation_fix @ RT
        Path(out_folder / f"{idx:05d}.txt").write_text(f"""{RT[0, 0]} {RT[0, 1]} {RT[0, 2]} {RT[0, 3]}\n{RT[1, 0]} {RT[1, 1]} {RT[1, 2]} {RT[1, 3]}\n{RT[2, 0]} {RT[2, 1]} {RT[2, 2]} {RT[2, 3]}\n0.00 0.00 0.00 1.00""")
    K = np.array([[float(y.strip()) for y in x.strip().split()] for x in Path(src_folder / "intrinsic" / "intrinsic_color.txt").read_text().splitlines() if x != ''])
    K = K[:3, :3]
    K = np.diag([640 / img_w, 480 / img_h, 1]) @ K
    Path(out_folder, "intrinsic.txt").write_text(f"""{K[0][0]} {K[0][1]} {K[0][2]} 0.00\n{K[1][0]} {K[1][1]} {K[1][2]} 0.00\n{K[2][0]} {K[2][1]} {K[2][2]} 0.00\n0.00 0.00 0.00 1.00""")
    out_folder = mmdet_folder / "data" / "scannet" / "scans" / src_folder.stem
    out_folder.mkdir(exist_ok=True)
    if (from_replica_github / src_folder.stem / "mesh.ply").exists():
        shutil.copyfile(from_replica_github / src_folder.stem / "mesh.ply", out_folder / f"{src_folder.stem}_vh_clean_2.ply")


def mmdet_create_gt_bboxes(src_folder):
    sc_classname = get_classnames()
    is_thing = get_thing_semantics()
    replica_to_scannet = get_replica_to_scannet(src_folder, sc_classname)
    pkl_segmentation_data = pickle.load(open(src_folder / f'segmentation_data.pkl', 'rb'))
    plydata = plyfile.PlyData.read(str(from_replica_github / src_folder.stem / "habitat/mesh_semantic.ply"))
    vertex_data = np.array([e.tolist() for e in plydata["vertex"]])
    vertices = vertex_data[:, :3]
    face_indices = np.array([e.tolist()[0] for e in plydata["face"]])
    class_face_ids = np.array([e.tolist()[1] for e in plydata["face"]])
    used_class_ids = np.unique(class_face_ids)
    valid_bbox_id = 0
    bboxes = {}
    distinct_colors = DistinctColors()

    rotation_fix = np.eye(4)
    if src_folder.stem in scene_specific_fixes_rotation:
        axangle = scene_specific_fixes_rotation[src_folder.stem]
        rotation_fix[:3, :3] = axangle2mat(axangle[1:4], axangle[0])

    tmesh = trimesh.load(from_replica_github / src_folder.stem / "mesh.ply")
    tmesh = tmesh.apply_transform(hmg(rotation_fix))
    translation_fix = hmg(np.eye(3))
    translation_fix[0, 3] = -(tmesh.bounds[0][0] + tmesh.bounds[1][0]) * 0.5
    translation_fix[1, 3] = -(tmesh.bounds[0][1] + tmesh.bounds[1][1]) * 0.5
    translation_fix[2, 3] = -tmesh.bounds[0][2]

    (src_folder / "visualized_gtboxes").mkdir(exist_ok=True)
    for current_class_id in tqdm(used_class_ids):
        class_label = replica_to_scannet[current_class_id].item()
        # is_thing[class_label] = True
        if is_thing[class_label]:
            current_face_indices = face_indices[class_face_ids == current_class_id]
            current_face_indices = current_face_indices.reshape(-1)
            current_vertices = vertices[current_face_indices]
            position = (translation_fix @ rotation_fix @ np.concatenate([current_vertices.mean(0), [1]]))[:3]
            extent = (current_vertices.max(0) - current_vertices.min(0)) * 1.05
            orientation = np.eye(3)
            bboxes[valid_bbox_id] = {
                'position': position,
                'orientation': orientation,
                'extent': extent,
                'class': class_label
            }
            # visualize_points(current_vertices, src_folder / "visualized_gtboxes" / f"pc_{sc_classname[class_label]}_{valid_bbox_id}.obj")
            # mesh = trimesh.Trimesh(vertices=current_vertices, faces=np.array(list(range(current_vertices.shape[0]))).reshape(-1, 4))
            # mesh = mesh.simplify_quadratic_decimation(face_count=50)
            # mesh.export(src_folder / "visualized_gtboxes" / f"pc_{sc_classname[class_label]}_{valid_bbox_id}.obj")
            create_box(bboxes[valid_bbox_id]['position'], bboxes[valid_bbox_id]['extent'], bboxes[valid_bbox_id]['orientation'], distinct_colors.get_color_fast_numpy(class_label)).export(src_folder / "visualized_gtboxes" / f"{sc_classname[class_label]}_{valid_bbox_id}.obj")
            valid_bbox_id += 1
    pkl_segmentation_data['gt_bboxes'] = bboxes
    pickle.dump(pkl_segmentation_data, open(src_folder / f'segmentation_data.pkl', 'wb'))


def create_m2f_consistent_instances(src_folder):
    instance_folder = src_folder / "m2f_notta_instance"
    semantics_folder = src_folder / "m2f_notta_semantics"
    gt_instance_folder = src_folder / "rs_instance"
    gt_semantics_folder = src_folder / "rs_semantics"
    output_folder = src_folder / "m2f_notta_instance_correspondences"
    output_folder.mkdir(exist_ok=True)
    unique_gt_instance_ids = []
    for f in tqdm(list(instance_folder.iterdir()), desc='creating instance renumbering'):
        instance_gt = np.array(Image.open(gt_instance_folder / f.name))
        [unique_gt_instance_ids.append(int(x)) for x in np.unique(instance_gt)]
    unique_gt_instance_ids = sorted(list(set(unique_gt_instance_ids)))
    for f in tqdm(list(instance_folder.iterdir()), desc='creating new mask'):
        instance = np.array(Image.open(f))
        instance_gt = np.array(Image.open(gt_instance_folder / f.name))
        semantics = np.array(Image.open(semantics_folder / f.name))
        semantics_gt = np.array(Image.open(gt_semantics_folder / f.name))
        new_instance = np.zeros_like(instance)
        classes = np.unique(instance)
        classes = classes[classes != 0]
        for c in classes:
            mask = np.logical_and(semantics_gt == mode(semantics[instance == c]).mode[0], instance == c)
            uniques, counts = np.unique(instance_gt[mask], return_counts=True)
            uniques, counts = uniques[uniques != 0], counts[uniques != 0]
            if len(counts) > 0:
                assigned_index = unique_gt_instance_ids.index(int(uniques[counts.argmax()]))
                new_instance[instance == c] = assigned_index
        Image.fromarray(new_instance).save(output_folder / f.name)


def copy_to_dm_nerf(src_folder):
    dm_nerf_path = Path("/cluster/gimli/ysiddiqui/dm-nerf-data/replica") / src_folder.stem
    shutil.copyfile(src_folder / "splits.json", dm_nerf_path / "splits.json")
    for suffix in ["", "_no_correspondences"]:
        output_folder = dm_nerf_path / f"semantic_instance_m2f{suffix}"
        output_folder.mkdir(exist_ok=True)
        input_folder = src_folder / f"m2f_notta_dmnerf{suffix}"
        output_names = sorted([x for x in list(Path(dm_nerf_path, "semantic_instance").iterdir()) if x.name.startswith('semantic')], key=lambda x: int(x.stem.split('_')[2]))
        input_names = sorted(list(input_folder.iterdir()), key=lambda x: int(x.stem.split('_')[1]))
        for idx in range(len(input_names)):
            shutil.copyfile(input_names[idx], output_folder / f"{output_names[idx].stem}.png")


def convert_ours_to_replica_pose_blender(src_folder):
    dm_nerf_path = Path("/cluster/gimli/ysiddiqui/dm-nerf-data/replica") / src_folder.stem
    poses_replica = []
    for line in (dm_nerf_path / "traj_w_c.txt").read_text().splitlines():
        poses_replica.append(np.array([float(x) for x in line.strip().split(" ")]).reshape((4, 4)))
    poses_ours = []
    for item in sorted(list((src_folder / "pose").iterdir()), key=lambda x: x.stem):
        RT = np.array([[float(y.strip()) for y in x.strip().split()] for x in Path(item).read_text().splitlines() if x != ''])
        poses_ours.append(RT)
    assert len(poses_replica) == len(poses_ours), "num poses not equal"
    X = []
    for i in range(len(poses_ours)):
        X.append((poses_replica[i] @ np.linalg.inv(poses_ours[i]))[np.newaxis, :, :])
    X = np.concatenate(X, 0).mean(0)
    traj_blender_new = ""
    for line in (dm_nerf_path / "traj_blender.txt").read_text().splitlines():
        RT = X @ np.array([float(x) for x in line.strip().split(" ")]).reshape((4, 4))
        traj_blender_new += f"""{RT[0, 0]} {RT[0, 1]} {RT[0, 2]} {RT[0, 3]} {RT[1, 0]} {RT[1, 1]} {RT[1, 2]} {RT[1, 3]} {RT[2, 0]} {RT[2, 1]} {RT[2, 2]} {RT[2, 3]} 0.00 0.00 0.00 1.00\n"""
    (dm_nerf_path / "traj_blender_new.txt").write_text(traj_blender_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='replica preprocessing')
    parser.add_argument('--root_path', required=False, default='data/replica/office_3', help='sens file path')
    parser.add_argument('--n', required=False, type=int, default=1, help='num proc')
    parser.add_argument('--p', required=False, type=int, default=0, help='current proc')
    args = parser.parse_args()

    dest = Path(args.root_path)
    # copy image from raw folder to root dataset
    copy_images_to_src_folder(dest)
    # copy cameras from raw folder to root dataset
    copy_cam_to_src_folder(dest)
    # copy GT annotations
    extract_semantics_and_instances(dest)
    # map gt to scannet classes
    map_gt_to_scannet(dest)
    # visualize annotations
    visualize_mask_folder(dest / "semantics")
    visualize_mask_folder(dest / "instance")
    visualize_mask_folder(dest / "rs_semantics")
    visualize_mask_folder(dest / "rs_instance")
    # create validation set
    create_validation_set(dest, 0.25)
    # make sure that mask2former has been run to create panoptic labels folder before proceeding
    # map mask2former labels
    map_panoptic_coco(dest)
    # visualize labels
    visualize_mask_folder(dest / "m2f_semantics")
    visualize_mask_folder(dest / "m2f_instance")
    visualize_mask_folder(dest / "m2f_notta_semantics")
    visualize_mask_folder(dest / "m2f_notta_instance")
    visualize_mask_folder(dest / "m2f_segments")
