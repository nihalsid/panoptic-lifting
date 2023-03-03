# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from pathlib import Path
from pylab import *
import argparse
import csv
import json
import shutil
from os import path
from scipy import stats

import h5py
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

from dataset.preprocessing.preprocess_scannet import get_reduce_and_fold_map, get_thing_semantics, visualize_mask_folder, map_panoptic_coco, create_validation_set
from util.camera import distance_to_depth
from util.distinct_colors import DistinctColors
from util.misc import create_box, visualize_mask
from util.transforms import trans_mat, hmg, trs_decomp, trs_comp

preview_path = Path("/rhome/ysiddiqui/ml-hypersim/contrib/99991/downloads")
multiframe_path = Path("/rhome/ysiddiqui/ml-hypersim/contrib/99991/multiframe")
raw_path = Path("/cluster/gimli/ysiddiqui/nerf-lightning-data/hypersim/raw")

percentile_brightness = {
    "ai_001_003": 0.8,
    "ai_004_006": 0.7
}

scene_specific_fixes_translation = {
    "ai_001_003": [-(-5.4810 + 3.7987) / 2, -(-7.9913 + 1.0447) / 2, 0],
    "ai_001_006": [-(-1.8426 + 4.9771) / 2, -(-6.4138 + 0.4203) / 2, 0],
    "ai_001_008": [-(-4.0580 + 4.9945) / 2, -(-6.4906 + 0.3242) / 2, 0],
    "ai_008_004": [-(-1.3509 + 1.9038) / 2, -(-2.2970 + 0.7725) / 2, 0],
    "ai_035_001": [-(2.7915 + 7.7338) / 2, -(-3.2882 + 2.0446) / 2, 0],
    "ai_048_008": [0, 0, 4.2708],
}

# fix ambigous or incorrect labels
scene_specific_fixes_objectid = {
    "ai_001_003": {
        20: 25,
        61: 3,
        103: 9
    },
    "ai_001_006": {
        7: 7,
        8: 7,
        25: 9,
        26: 9,
        28: 9,
        31: 3
    },
    "ai_001_008": {
        2: 6,
        1: 6,
        4: 6,
        40: 9,
        42: 9,
        43: 9,
        44: 9,
        45: 9,
    },
    "ai_001_010": {
        23: 7
    },
    "ai_010_005": {
        2: 3,
        4: 3,
        5: 3,
        6: 3
    },
    "ai_035_001": {
        13: 16,
        14: 16,
        22: 3,
        34: 3
    },
    "ai_048_008": {
        1: 9,
        3: 9,
        7: 6,
        21: 7,
        36: 6,
        50: 7
    },
    "ai_027_008": {
        39: 7
    }
}


def export_hypersim_for_manual_inspection():
    export_folder = preview_path.parent / "preview"
    export_folder.mkdir(exist_ok=True)
    for folder in preview_path.iterdir():
        shutil.copyfile(folder / "images" / "scene_cam_00_final_preview" / "frame.0000.color.jpg", export_folder / (folder.name + ".jpg"))


def export_multiview_hypersim_for_manual_inspection():
    export_folder = preview_path.parent / "preview_multiview"
    export_folder.mkdir(exist_ok=True)
    for folder in sorted(multiframe_path.iterdir()):
        if (folder / "images" / "scene_cam_00_final_preview" / "frame.0000.color.jpg").exists():
            shutil.copyfile(preview_path / folder.name / "images" / "scene_cam_00_final_preview" / "frame.0000.color.jpg", export_folder / (folder.name + "_0.jpg"))
        if (folder / "images" / "scene_cam_01_final_preview" / "frame.0000.color.jpg").exists():
            shutil.copyfile(folder / "images" / "scene_cam_01_final_preview" / "frame.0000.color.jpg", export_folder / (folder.name + "_1.jpg"))
        if (folder / "images" / "scene_cam_01_final_preview" / "frame.0005.color.jpg").exists():
            shutil.copyfile(folder / "images" / "scene_cam_01_final_preview" / "frame.0005.color.jpg", export_folder / (folder.name + "_2.jpg"))


def tone_map_hdf5(src_folder):
    raw_src = raw_path / src_folder.stem / "images"
    cam_folders = [x for x in raw_src.iterdir() if x.name.endswith("final_hdf5")]
    for cf in cam_folders:
        rf = cf.parent / (cf.name.split("final_hdf5")[0] + "geometry_hdf5")
        outfolder = cf.parent / (cf.name.split("final_hdf5")[0] + "final_tone")
        outfolder.mkdir(exist_ok=True)
        for in_rgb_hdf5_file in tqdm([x for x in cf.iterdir() if x.name.endswith('.color.hdf5')]):
            with h5py.File(in_rgb_hdf5_file, "r") as f:
                rgb_color = f["dataset"][:].astype(np.float32)
            scale = 1
            always_render_same = True
            gamma = 1.0 / (2.2 * 1)  # standard gamma correction exponent
            if not always_render_same:
                inv_gamma = 1.0 / gamma
                percentile = 90  # we want this percentile brightness value in the unmodified image...
                brightness_nth_percentile_desired = percentile_brightness[src_folder.stem]  # ...to be this bright after scaling
                with h5py.File(rf / (in_rgb_hdf5_file.name.split('.color.hdf5')[0] + ".render_entity_id.hdf5"), "r") as f:
                    render_entity_id = f["dataset"][:].astype(np.int32)
                valid_mask = render_entity_id != -1
                if count_nonzero(valid_mask) != 0:
                    brightness = 0.3 * rgb_color[:, :, 0] + 0.59 * rgb_color[:, :, 1] + 0.11 * rgb_color[:, :, 2]  # "CCIR601 YIQ" method for computing brightness
                    brightness_valid = brightness[valid_mask]
                    eps = 0.0001  # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
                    brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)
                    if brightness_nth_percentile_current < eps:
                        scale = 0.0
                    else:
                        scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current
            rgb_color_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
            rgb_color_tm = clip(rgb_color_tm, 0, 1)
            # rgb_color_tm = np.array(Image.fromarray((rgb_color_tm * 255).astype(np.uint8)).resize((640, 480)))
            imsave(outfolder / (in_rgb_hdf5_file.stem + ".jpg"), rgb_color_tm)


CAMERA_PARAMS_FILE = "metadata_camera_parameters.csv"
R_hc = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])


def extract_hypersim(data_dir, seq_id, output_dir):
    seq_dir = path.join(data_dir, seq_id)

    # Load the scene params
    fx, fy, cx, cy = load_camera_params(data_dir, seq_id)
    meters_per_unit = load_meters_per_unit(data_dir, seq_id)

    # Find the cameras
    cameras = []
    with open(path.join(seq_dir, "_detail", "metadata_cameras.csv"), "r") as fid:
        for i, line in enumerate(fid):
            if i == 0:
                continue
            camera_id = line.strip()

            # Verify that the data for this camera is available
            if os.path.exists(path.join(seq_dir, "_detail", camera_id)):
                cameras.append(camera_id)

    # Output paths
    out_img_dir = path.join(output_dir, "color")
    out_msk_dir = path.join(output_dir, "mask")
    out_msk_instance_dir = path.join(output_dir, "mask_instance")
    out_dpt_dir = path.join(output_dir, "depth_npy")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)
    os.makedirs(out_msk_instance_dir, exist_ok=True)
    os.makedirs(out_dpt_dir, exist_ok=True)

    metadata = {"calibration": {"fx": fx, "fy": fy, "cx": cx, "cy": cy}, "images": []}

    instances = {}
    extents_file = "metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5"
    orientations_file = "metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5"
    positions_file = "metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5"
    h5handle_extents = h5py.File(path.join(seq_dir, "_detail", "mesh", extents_file), "r")
    h5handle_orientations = h5py.File(path.join(seq_dir, "_detail", "mesh", orientations_file), "r")
    h5handle_positions = h5py.File(path.join(seq_dir, "_detail", "mesh", positions_file), "r")
    instance_count = h5handle_positions['dataset'].shape[0]
    for k in range(instance_count):
        instances[k] = {
            "extent": np.array(h5handle_extents['dataset'][k]).astype(np.float32) * meters_per_unit,
            "orientation": np.array(h5handle_orientations['dataset'][k]).astype(np.float32),
            "position": np.array(h5handle_positions['dataset'][k]).astype(np.float32) * meters_per_unit
        }
    with open(path.join(output_dir, "instances.pkl"), "wb") as fid:
        pickle.dump(instances, fid)

    img_counter = 0
    for camera_id in cameras:
        # Relevant input paths
        img_dir = path.join(seq_dir, "images", f"scene_{camera_id}_final_tone")
        msk_dir = path.join(seq_dir, "images", f"scene_{camera_id}_geometry_hdf5")
        cam_dir = path.join(seq_dir, "_detail", camera_id)

        # Load camera positions and rotations
        cam_pos, cam_rot = load_camera_pos_and_rot(cam_dir)

        # Copy images and export data
        for i, (cam_pos_i, cam_rot_i) in tqdm(
            enumerate(zip(cam_pos, cam_rot)), total=cam_pos.shape[0]
        ):
            # Verify that all files are available
            img_file = path.join(img_dir, f"frame.{i:04d}.color.jpg")
            msk_file = path.join(msk_dir, f"frame.{i:04d}.semantic.hdf5")
            msk_instance_file = path.join(msk_dir, f"frame.{i:04d}.semantic_instance.hdf5")
            dpt_file = path.join(msk_dir, f"frame.{i:04d}.depth_meters.hdf5")
            for f in [img_file, msk_file, msk_instance_file, dpt_file]:
                if not os.path.exists(f):
                    print("WARNING: ", f)
                    exit()

            # Copy the image
            with open(img_file, "rb") as in_fid, open(
                path.join(out_img_dir, f"{img_counter:04d}.jpg"), "wb"
            ) as out_fid:
                shutil.copyfileobj(in_fid, out_fid)

            # Copy the semantic mask
            with open(msk_file, "rb") as in_fid, open(
                path.join(out_msk_dir, f"{img_counter:04d}.png"), "wb"
            ) as out_fid:
                msk = h5py.File(in_fid, "r")
                msk = np.array(msk["dataset"], copy=True)

                # Void: -1 -> 0
                msk[msk == -1] = 0
                Image.fromarray(msk.astype(np.uint8)).save(out_fid)

            # Copy the instance mask
            with open(msk_instance_file, "rb") as in_fid, open(
                    path.join(out_msk_instance_dir, f"{img_counter:04d}.png"), "wb"
            ) as out_fid:
                msk = h5py.File(in_fid, "r")
                msk = np.array(msk["dataset"], copy=True)
                # Void: -1 -> 255
                msk[msk == -1] = 0
                Image.fromarray(msk.astype(np.uint8)).save(out_fid)

            # Copy the depth
            with open(dpt_file, "rb") as in_fid, open(
                path.join(out_dpt_dir, f"{img_counter:04d}.npy"), "wb"
            ) as out_fid:
                dpt = h5py.File(in_fid, "r")
                dpt = np.array(dpt["dataset"], copy=True)
                np.save(out_fid, dpt.astype(np.float32))

            # Convert pos and rot to standard format
            pos, rot = convert_camera_pose(cam_pos_i, cam_rot_i)
            pos = pos * meters_per_unit

            metadata["images"].append(
                {"rotation": rot.tolist(), "translation": pos.tolist()}
            )

            img_counter += 1

    # Save the metadata
    with open(path.join(output_dir, "metadata.json"), "w") as fid:
        json.dump(metadata, fid)


def load_camera_params(data_dir, seq_id):
    # Load all the sequence parameters
    seq_data = None
    with open(path.join(data_dir, CAMERA_PARAMS_FILE), newline="") as fid:
        reader = csv.reader(fid)
        for i, line in enumerate(reader):
            if i == 0:
                columns = line
            else:
                if line[0] == seq_id:
                    seq_data = {c: v for c, v in zip(columns[1:], line[1:])}

    if seq_data is None:
        print(f"Unable to find sequence {seq_id} in the camera parameters file")
        exit(1)

    height = float(seq_data["settings_output_img_height"])
    width = float(seq_data["settings_output_img_width"])
    a = float(seq_data["M_proj_00"])
    b = float(seq_data["M_proj_11"])

    fx = 0.5 * a * (width - 1)
    fy = 0.5 * b * (height - 1)
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)

    return fx, fy, cx, cy


def load_meters_per_unit(data_dir, seq_id):
    with open(
        path.join(data_dir, seq_id, "_detail", "metadata_scene.csv"), "r"
    ) as fid:
        for line in fid:
            toks = line.strip().split(",")
            if toks[0] == "meters_per_asset_unit":
                return float(toks[1])

    raise IOError(f"Unable to read the meters_per_unit value for sequence {seq_id}")


def load_camera_pos_and_rot(cam_dir):
    with open(
        path.join(cam_dir, "camera_keyframe_positions.hdf5"), "rb"
    ) as fid:
        data = h5py.File(fid, "r")
        cam_pos = np.array(data["dataset"], copy=True)

    with open(
        path.join(cam_dir, "camera_keyframe_orientations.hdf5"), "rb"
    ) as fid:
        data = h5py.File(fid, "r")
        cam_rot = np.array(data["dataset"], copy=True)

    return cam_pos, cam_rot


def convert_camera_pose(pos, rot):
    out_rot = np.dot(R_hc, rot.T)
    out_pos = -np.dot(out_rot, pos)
    return out_pos, out_rot


def fold_nyu_classes(src_folder):
    instance_to_semantics = {}
    reduce_map, fold_map = get_reduce_and_fold_map()
    thing_classes = get_thing_semantics()
    stuff_classes = [i for i in range(len(thing_classes)) if thing_classes[i] == False]
    output_folder_sem = src_folder / "rs_semantics"
    output_folder_ins = src_folder / "rs_instance"
    output_folder_sem.mkdir(exist_ok=True)
    output_folder_ins.mkdir(exist_ok=True)

    for f in tqdm((src_folder / "mask").iterdir(), desc='folding semantics'):
        sem_arr = np.array(Image.open(f))
        ins_arr = np.array(Image.open(src_folder / "mask_instance" / f.name))
        if src_folder.stem in scene_specific_fixes_objectid:
            for ob_id in scene_specific_fixes_objectid[src_folder.stem]:
                sem_arr[ins_arr == ob_id] = scene_specific_fixes_objectid[src_folder.stem][ob_id]
        shape = sem_arr.shape
        folded_semantics = fold_map[reduce_map[sem_arr.flatten()]].reshape(shape).astype(np.int8)
        Image.fromarray(folded_semantics).save(output_folder_sem / f.name)
        ins_arr[np.isin(folded_semantics, stuff_classes)] = 0
        for u in np.unique(ins_arr):
            if u != 0:
                c = stats.mode(folded_semantics[ins_arr == u])[0][0]
                instance_to_semantics[u] = c
        Image.fromarray(ins_arr).save(output_folder_ins / f.name)
    print("rs_instance_to_semantic", instance_to_semantics)
    return instance_to_semantics


def get_translation_fix(src_folder):
    translation_fix = hmg(np.eye(3))
    if src_folder.stem in scene_specific_fixes_translation:
        tfix = scene_specific_fixes_translation[src_folder.stem]
        translation_fix[0, 3] = tfix[0]
        translation_fix[1, 3] = tfix[1]
        translation_fix[2, 3] = tfix[2]
    return translation_fix


def hypersim_export_to_scannet(src_folder):
    world2scene = None
    scene_name = src_folder.stem
    metadata = json.load(open(src_folder / "metadata.json"))
    calib = metadata["calibration"]
    K = np.array([[calib["fx"], 0, calib["cx"]], [0, calib["fy"], calib["cy"]], [0, 0, 1]])
    scene_max_depth = 0

    Path(src_folder, "intrinsic").mkdir(exist_ok=True)
    Path(src_folder, "pose").mkdir(exist_ok=True)
    Path(src_folder, "depth").mkdir(exist_ok=True)

    # convert sem to reduced sem
    instance_to_semantic = fold_nyu_classes(src_folder)
    translation_fix = get_translation_fix(src_folder)

    for sample_idx, cam_data in enumerate(metadata["images"]):
        # skip already created if resumed
        world2cam = trans_mat(cam_data["translation"]) @ hmg(np.array(cam_data["rotation"]))

        cam2world = translation_fix @ np.linalg.inv(world2cam)
        rotation_scale = np.abs(trs_decomp(cam2world)[-1][0]-1)
        if rotation_scale > 1e-2:
            print(f"Scale in rotation: {scene_name}, {sample_idx}  -- {rotation_scale} ")
            t, r, s = trs_decomp(cam2world)
            cam2world = trs_comp(t, r, np.eye(3))

        dpt = np.load(Path(src_folder, "depth_npy", f"{sample_idx:04}.npy"))
        depth = distance_to_depth(K, dpt).reshape((dpt.shape[0], dpt.shape[1])).reshape((dpt.shape[0], dpt.shape[1]))

        scene_max_depth = max(scene_max_depth, float(depth.max()))
        Path(src_folder, "max_depth.txt").write_text(f"{scene_max_depth}")

        # if sample_idx == 0:  # first cam frame defines scene frame
        #     world2scene = world2cam

        # cam2world = world2scene @ cam2world
        Path(src_folder, "intrinsic", "intrinsic_color.txt").write_text(f"""{K[0][0]} {K[0][1]} {K[0][2]} 0.00\n{K[1][0]} {K[1][1]} {K[1][2]} 0.00\n{K[2][0]} {K[2][1]} {K[2][2]} 0.00\n0.00 0.00 0.00 1.00""")
        Path(src_folder, "pose", f"{sample_idx:04}.txt").write_text(f"""{cam2world[0, 0]} {cam2world[0, 1]} {cam2world[0, 2]} {cam2world[0, 3]}\n{cam2world[1, 0]} {cam2world[1, 1]} {cam2world[1, 2]} {cam2world[1, 3]}\n{cam2world[2, 0]} {cam2world[2, 1]} {cam2world[2, 2]} {cam2world[2, 3]}\n0.00 0.00 0.00 1.00""")

        # convert depth to image
        Image.fromarray((depth * 1000).astype(np.int16)).save(src_folder / "depth" / f"{sample_idx:04}.png")

    thing_semantics = get_thing_semantics()
    if (src_folder / 'segmentation_data.pkl').exists():
        export_dict = pickle.load(open(src_folder / 'segmentation_data.pkl', 'rb'))
    else:
        export_dict = {}
    export_dict['num_semantic_classes'] = len(thing_semantics)
    export_dict['fg_classes'] = [i for i, is_thing in enumerate(thing_semantics) if is_thing]
    export_dict['bg_classes'] = [i for i, is_thing in enumerate(thing_semantics) if not is_thing]
    instance_to_semantic[0] = 0
    export_dict[f'rs_instance_to_semantic'] = instance_to_semantic
    # save bboxes
    pickle.dump(export_dict, open(src_folder / 'segmentation_data.pkl', 'wb'))
    map_gt_bboxes(dest)


def map_gt_bboxes(src_folder):
    distinct_colors = DistinctColors()
    translation_fix = get_translation_fix(src_folder)
    bboxes = {}
    valid_boxid = 0
    (src_folder / "visualized_gtboxes").mkdir(exist_ok=True)
    pkl_segmentation_data = pickle.load(open(src_folder / f'segmentation_data.pkl', 'rb'))
    bbox_annot = pickle.load(open(src_folder / f'instances.pkl', 'rb'))

    for bbox_idx in bbox_annot.keys():
        if bbox_idx in pkl_segmentation_data['rs_instance_to_semantic'] and bbox_idx != 0:
            label = pkl_segmentation_data['rs_instance_to_semantic'][bbox_idx]
            bboxes[valid_boxid] = {
                'position': bbox_annot[bbox_idx]["position"] + translation_fix[:3, 3],
                'orientation': bbox_annot[bbox_idx]["orientation"],
                'extent': bbox_annot[bbox_idx]["extent"],
                'class': label
            }
            create_box(bbox_annot[bbox_idx]["position"], bbox_annot[bbox_idx]["extent"], bbox_annot[bbox_idx]["orientation"], distinct_colors.get_color_fast_numpy(label)).export(src_folder / "visualized_gtboxes" / f"{label}_{valid_boxid}.obj")
            valid_boxid += 1
    pkl_segmentation_data['gt_bboxes'] = bboxes
    pickle.dump(pkl_segmentation_data, open(src_folder / f'segmentation_data.pkl', 'wb'))


def debug_dump_instances_for_replica_scene(mask_path):
    instance = np.array(Image.open(mask_path))
    u, c = np.unique(instance, return_counts=True)
    for uin in u:
        visualize_mask((instance == uin).astype(int), f"inst_{uin}.png")


def copy_to_dm_nerf(src_folder):
    dm_nerf_path = Path("/cluster/gimli/ysiddiqui/dm-nerf-data/hypersim") / src_folder.stem
    dm_nerf_path.mkdir(exist_ok=True)
    shutil.copyfile(src_folder / "splits.json", dm_nerf_path / "splits.json")
    for suffix in ["_no_correspondences"]:
        output_folder = dm_nerf_path / f"semantic_instance_m2f{suffix}"
        if output_folder.exists():
            shutil.rmtree(output_folder)
        output_folder.mkdir()
        input_folder = src_folder / f"m2f_notta_dmnerf{suffix}"
        input_names = sorted(list(input_folder.iterdir()), key=lambda x: int(x.stem))
        output_names = [f"semantic_instance_{int(x.stem)}" for x in input_names]
        for idx in range(len(input_names)):
            shutil.copyfile(input_names[idx], output_folder / f"{output_names[idx]}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypersim preprocessing')
    parser.add_argument('--root_path', required=False, default='data/hypersim/ai_001_008', help='sens file path')
    args = parser.parse_args()
    dest = Path(args.root_path)
    dest.mkdir(exist_ok=True)
    (dest / "panoptic").mkdir(exist_ok=True)
    # tone map raw data from hypersim
    tone_map_hdf5(dest)
    extract_hypersim(raw_path, dest.stem, dest)
    # make sure to run mask2former on rgb images before proceeding this step
    # GT scannet labels
    hypersim_export_to_scannet(dest)
    # visualize GT
    visualize_mask_folder(dest / "rs_semantics")
    visualize_mask_folder(dest / "rs_instance")
    # map mask2former labels
    map_panoptic_coco(dest)
    # visualize mask2former labels
    visualize_mask_folder(dest / "m2f_notta_semantics")
    visualize_mask_folder(dest / "m2f_semantics")
    visualize_mask_folder(dest / "m2f_notta_instance")
    visualize_mask_folder(dest / "m2f_instance")
    create_validation_set(dest, 0.15)
