# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import pickle
import shutil
import cv2
from PIL import Image

from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import os

from dataset.preprocessing.preprocess_scannet import create_validation_set, map_panoptic_coco, visualize_mask_folder, get_keyframe_indices, \
    from_ours_to_replica_traj_w_c, create_instances_for_dmnerf, get_thing_semantics

raw_path = Path("/cluster/gimli/ysiddiqui/nerf-lightning-data/itw/raw/")


def copy_color(src_folder, fraction):
    Path(src_folder, "color_distorted").mkdir(exist_ok=True, parents=True)
    Path(src_folder, "panoptic").mkdir(exist_ok=True, parents=True)
    color_folder = raw_path / src_folder.stem / "color"
    image_files = sorted(list(color_folder.iterdir()), key=lambda x: int(x.stem) if x.stem.isnumeric() else x.stem)
    frame_indices, _ = get_keyframe_indices(image_files, fraction)
    for i in tqdm(frame_indices):
        shutil.copyfile(image_files[i], src_folder / "color_distorted" / f'{i:04d}.jpg')


def rename_and_copy_transforms(src_folder):
    transform_file = raw_path / src_folder.stem / "transforms.json"
    transforms = json.loads(Path(transform_file).read_text())
    color_folder = raw_path / src_folder.stem / "color"
    image_files = sorted([x.stem for x in color_folder.iterdir()], key=lambda x: int(x) if x.isnumeric() else x)
    for idx, frame in enumerate(transforms['frames']):
        transforms['frames'][idx]['file_path'] = f"{image_files.index(Path(frame['file_path']).stem):04d}.jpg"
    (src_folder / "transforms.json").write_text(json.dumps(transforms))


def create_poses_without_undistortion(src_folder):
    color_folder = src_folder / "color_distorted"
    image_files = sorted(list(color_folder.iterdir()), key=lambda x: int(x.stem))
    if Path(src_folder, "color").exists():
        shutil.rmtree(Path(src_folder, "color"))
    if Path(src_folder, "panoptic").exists():
        shutil.rmtree(Path(src_folder, "panoptic"))
    Path(src_folder, "color").mkdir()
    Path(src_folder, "panoptic").mkdir()
    transforms = json.loads((src_folder / "transforms.json").read_text())
    for idx, image_file in enumerate(tqdm(image_files)):
        Image.open(image_file).save(Path(src_folder, "color") / image_file.name)
    h, w, cx, cy = int(transforms["h"]), int(transforms["w"]), transforms["cx"], transforms["cy"]
    mtx = np.array([[transforms["fl_x"], 0, cx], [0, transforms["fl_y"], cy], [0, 0, 1]])
    Path(src_folder, "intrinsic").mkdir(exist_ok=True)
    Path(src_folder, "intrinsic", "intrinsic_color.txt").write_text(
        f"""{mtx[0, 0]} {mtx[0, 1]} {mtx[0, 2]} 0.00\n{mtx[1, 0]} {mtx[1, 1]} {mtx[1, 2]} 0.00\n{mtx[2, 0]} {mtx[2, 1]} {mtx[2, 2]} 0.00\n0.00 0.00 0.00 1.00""")

    Path(src_folder, "pose").mkdir(exist_ok=True)
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    for frame in transforms['frames']:
        filepath = Path(frame['file_path'])
        RT = np.array(frame['transform_matrix']) @ flip_mat
        Path(src_folder, "pose", f'{filepath.stem}.txt').write_text(f"""{RT[0, 0]} {RT[0, 1]} {RT[0, 2]} {RT[0, 3]}\n{RT[1, 0]} {RT[1, 1]} {RT[1, 2]} {RT[1, 3]}\n{RT[2, 0]} {RT[2, 1]} {RT[2, 2]} {RT[2, 3]}\n0.00 0.00 0.00 1.00""")

    pose_files = [x.stem for x in (src_folder / "pose").iterdir()]
    image_files = [x.stem for x in (src_folder / "color").iterdir()]
    to_remove = list(set(image_files) - set(pose_files))
    for f in to_remove:
        os.remove(src_folder / "color" / f"{f}.jpg")


def create_undistorted_images(src_folder):
    color_folder = src_folder / "color_distorted"
    image_files = sorted(list(color_folder.iterdir()), key=lambda x: int(x.stem))
    if Path(src_folder, "color").exists():
        shutil.rmtree(Path(src_folder, "color"))
    if Path(src_folder, "panoptic").exists():
        shutil.rmtree(Path(src_folder, "panoptic"))
    Path(src_folder, "color").mkdir()
    Path(src_folder, "panoptic").mkdir()
    all_images = []
    for image_file in tqdm(image_files, desc='read'):
        all_images.append(np.array(Image.open(image_file))[np.newaxis, :, :, :])
    all_images = np.concatenate(all_images, axis=0)
    transforms = json.loads((src_folder / "transforms.json").read_text())
    if "camera_model" in transforms and transforms["camera_model"] == "OPENCV_FISHEYE":
        h, w, cx, cy, k1, k2, k3, k4 = int(transforms["h"]), int(transforms["w"]), transforms["cx"], transforms["cy"], transforms["k1"], transforms["k2"], transforms["k3"], transforms["k4"]
        distortion_params = np.array([k1, k2, k3, k4])
        mtx = np.array([[transforms["fl_x"], 0, cx], [0, transforms["fl_y"], cy], [0, 0, 1]])
        newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, distortion_params, [w, h], np.eye(3), balance=1)
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(mtx, distortion_params, np.eye(3), newcameramtx, [w, h], cv2.CV_16SC2)
        mask = np.ones([h, w]).astype(np.uint8) * 255
        distorted_mask = 255 - ((cv2.remap(mask, mapx, mapy, interpolation=cv2.INTER_CUBIC) > 0) * 255).astype(np.uint8)
        roi = [0, 0, w - 1, h - 1]
    else:
        h, w, cx, cy, k1, k2, p1, p2 = int(transforms["h"]), int(transforms["w"]), transforms["cx"], transforms["cy"], transforms["k1"], transforms["k2"], transforms["p1"], transforms["p2"]
        mtx = np.array([[transforms["fl_x"], 0, cx], [0, transforms["fl_y"], cy], [0, 0, 1]])
        dist = np.array([[k1, k2, p1, p2]])
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        distorted_mask = None

    for idx, image_file in enumerate(tqdm(image_files, desc='undistort')):
        img = cv2.remap(all_images[idx, :, :, :], mapx, mapy, cv2.INTER_CUBIC)
        Image.fromarray(img[roi[1]: roi[3] + 1, roi[0]: roi[2] + 1, :]).save(Path(src_folder, "color") / image_file.name)
        if distorted_mask is not None:
            Path(src_folder, "invalid").mkdir(exist_ok=True)
            Image.fromarray(distorted_mask).save(Path(src_folder, "invalid") / image_file.name)

    Path(src_folder, "intrinsic").mkdir(exist_ok=True)
    Path(src_folder, "intrinsic", "intrinsic_color.txt").write_text(
        f"""{newcameramtx[0, 0]} {newcameramtx[0, 1]} {newcameramtx[0, 2]} 0.00\n{newcameramtx[1, 0]} {newcameramtx[1, 1]} {newcameramtx[1, 2]} 0.00\n{newcameramtx[2, 0]} {newcameramtx[2, 1]} {newcameramtx[2, 2]} 0.00\n0.00 0.00 0.00 1.00""")

    Path(src_folder, "pose").mkdir(exist_ok=True)
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    for frame in transforms['frames']:
        filepath = Path(frame['file_path'])
        RT = np.array(frame['transform_matrix']) @ flip_mat
        Path(src_folder, "pose", f'{filepath.stem}.txt').write_text(f"""{RT[0, 0]} {RT[0, 1]} {RT[0, 2]} {RT[0, 3]}\n{RT[1, 0]} {RT[1, 1]} {RT[1, 2]} {RT[1, 3]}\n{RT[2, 0]} {RT[2, 1]} {RT[2, 2]} {RT[2, 3]}\n0.00 0.00 0.00 1.00""")

    pose_files = [x.stem for x in (src_folder / "pose").iterdir()]
    image_files = [x.stem for x in (src_folder / "color").iterdir()]
    to_remove = list(set(image_files) - set(pose_files))
    for f in to_remove:
        os.remove(src_folder / "color" / f"{f}.jpg")


def export_for_semantic_nerf(src_folder):
    out_dir = src_folder.parent / "raw" / "from_semantic_nerf" / src_folder.name / "Sequence_1"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    # copy color -> rgb
    # remake splits
    splits = json.loads((src_folder / "splits.json").read_text())
    for split in ["train", "val"]:
        splits[split] = [f"{int(x):04d}" for x in splits[split]]
    Path(out_dir / "splits.json").write_text(json.dumps(splits))
    # copy intrinsics
    shutil.copyfile(src_folder / "intrinsic" / "intrinsic_color.txt", out_dir / "intrinsic_color.txt")
    # make trajectory and copy
    from_ours_to_replica_traj_w_c(src_folder)
    shutil.copyfile(src_folder / "traj_w_c.txt", out_dir / "traj_w_c.txt")
    (out_dir / "rgb").mkdir()
    for f in (src_folder / "color").iterdir():
        shutil.copyfile(f, out_dir / "rgb" / f"{int(f.stem):04d}.jpg")


def export_for_dmnerf(src_folder):
    dm_nerf_path = Path("/cluster/gimli/ysiddiqui/dm-nerf-data/scannet") / src_folder.name
    out_dir = dm_nerf_path
    if not out_dir.exists():
        shutil.copytree(src_folder.parent / "raw" / "from_semantic_nerf" / src_folder.name / "Sequence_1", out_dir)
    create_instances_for_dmnerf(src_folder, correspondences=False, class_set='extended')
    suffix = "_no_correspondences"
    output_folder = dm_nerf_path / f"semantic_instance_m2f{suffix}"
    output_folder.mkdir(exist_ok=True)
    input_folder = src_folder / f"m2f_notta_dmnerf{suffix}"
    input_names = sorted(list(input_folder.iterdir()), key=lambda x: int(x.stem))
    output_names = [f"semantic_instance_{int(x.stem)}" for x in input_names]
    for idx in range(len(input_names)):
        shutil.copyfile(input_names[idx], output_folder / f"{output_names[idx]}.png")


def create_segmentation_data(src_folder, sc_classes='reduced'):
    thing_semantics = get_thing_semantics(sc_classes)
    print('len thing_semantics', len(thing_semantics))
    export_dict = {
        'num_semantic_classes': len(thing_semantics),
        'fg_classes': [i for i, is_thing in enumerate(thing_semantics) if is_thing],
        'bg_classes': [i for i, is_thing in enumerate(thing_semantics) if not is_thing]
    }
    pickle.dump(export_dict, open(src_folder / 'segmentation_data.pkl', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='replica preprocessing')
    parser.add_argument('--root_path', required=False, default='data/itw/office_fisheye', help='sens file path')
    args = parser.parse_args()
    dest = Path(args.root_path)

    # copy distorted images from raw to root
    copy_color(dest, fraction=1)
    # copy transforms json
    rename_and_copy_transforms(dest)
    # create segmentation data stub
    create_segmentation_data(dest, sc_classes='extended')
    # undistort images
    create_undistorted_images(dest)
    # create_poses_without_undistortion(dest)
    # create validation set (15% to 25%)
    create_validation_set(dest, 0.15)
    # make sure to run mask2former before this step
    # run mask2former segmentation data mapping
    map_panoptic_coco(dest, sc_classes='extended')
    # visualize labels
    visualize_mask_folder(dest / "m2f_semantics")
    visualize_mask_folder(dest / "m2f_instance")
    visualize_mask_folder(dest / "m2f_notta_semantics")
    visualize_mask_folder(dest / "m2f_notta_instance")
    # copy predicted labels as GT, since we don't have GT
    shutil.copytree(dest / "m2f_semantics", dest / "semantics")
    shutil.copytree(dest / "m2f_instance", dest / "instance")
    shutil.copytree(dest / "m2f_semantics", dest / "rs_semantics")
    shutil.copytree(dest / "m2f_instance", dest / "rs_instance")
