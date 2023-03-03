import gzip
import json
import os
import pickle
import shutil
from collections import defaultdict, Counter
from pprint import pprint
import math
import numpy as np
import scipy
from scipy import stats
import argparse
from pathlib import Path

from matplotlib import cm
from transforms3d.axangles import axangle2mat
from util.transforms import hmg, dot
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from dataset.preprocessing.sens_reader.SensorData import SensorData
from util.distinct_colors import DistinctColors
from util.metrics import ConfusionMatrix
from util.misc import visualize_mask, create_box, get_boundary_mask
from util.panoptic_quality import panoptic_quality, panoptic_quality_match, _panoptic_quality_compute


def get_keyframe_indices(filenames, window_size):
    """
        select non-blurry images within a moving window
    """
    scores = []
    for filename in tqdm(filenames, 'processing keyframes'):
        img = cv2.imread(str(filename))
        blur_score = compute_blur_score_opencv(img)
        scores.append(blur_score)

    keyframes = [i + np.argmin(scores[i:i + window_size]) for i in range(0, len(scores), window_size)]
    return keyframes, scores


def compute_blur_score_opencv(image):
    """
    Estimate the amount of blur an image has with the variance of the Laplacian.
    Normalize by pixel number to offset the effect of image size on pixel gradients & variance
    https://github.com/deepfakes/faceswap/blob/ac40b0f52f5a745aa058f92339302065177dd28b/tools/sort/sort.py#L626
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_map = cv2.Laplacian(image, cv2.CV_32F)
    score = np.var(blur_map) / np.sqrt(image.shape[0] * image.shape[1])
    return 1.0 - score


def subsample_scannet(src_folder, rate):
    """
    sample every nth frame from scannet
    """
    all_frames = sorted(list(x.stem for x in (src_folder / 'pose').iterdir()), key=lambda y: int(y) if y.isnumeric() else y)
    total_sampled = int(len(all_frames) * rate)
    sampled_frames = [all_frames[i * (len(all_frames) // total_sampled)] for i in range(total_sampled)]
    unsampled_frames = [x for x in all_frames if x not in sampled_frames]
    for frame in sampled_frames:
        if 'inf' in Path(src_folder / "pose" / f"{frame}.txt").read_text():
            unsampled_frames.append(frame)
    folders = ["color", "depth", "instance", "pose", "semantics"]
    exts = ['jpg', 'png', 'png', 'txt', 'png']
    for folder, ext in tqdm(zip(folders, exts), desc='sampling'):
        assert (src_folder / folder).exists(), src_folder
        for frame in unsampled_frames:
            if (src_folder / folder / f'{frame}.{ext}').exists():
                os.remove(str(src_folder / folder / f'{frame}.{ext}'))
            else:
                print(str(src_folder / folder / f'{frame}.{ext}'), "already exists!")


def subsample_scannet_blur_window(src_folder, min_frames):
    """
    sample non blurry frames from scannet
    """
    all_frames = sorted(list(x.stem for x in (src_folder / 'pose').iterdir()), key=lambda y: int(y) if y.isnumeric() else y)
    all_frame_paths = sorted(list(x for x in (src_folder / 'color').iterdir()), key=lambda y: int(y.stem) if y.stem.isnumeric() else y.stem)
    if len(all_frame_paths) <= min_frames:
        sampled_frames = all_frames
    else:
        window_size = max(2, int(math.ceil(len(all_frames) / min_frames)))
        frame_indices, _ = get_keyframe_indices(all_frame_paths, window_size)
        print("Using a window size of", window_size, "got", len(frame_indices), "frames")
        sampled_frames = [all_frames[i] for i in frame_indices]
    unsampled_frames = [x for x in all_frames if x not in sampled_frames]
    for frame in sampled_frames:
        if 'inf' in Path(src_folder / "pose" / f"{frame}.txt").read_text():
            unsampled_frames.append(frame)
    folders = ["color", "depth", "instance", "pose", "semantics"]
    exts = ['jpg', 'png', 'png', 'txt', 'png']
    for folder, ext in tqdm(zip(folders, exts), desc='sampling'):
        assert (src_folder / folder).exists(), src_folder
        for frame in unsampled_frames:
            if (src_folder / folder / f'{frame}.{ext}').exists():
                os.remove(str(src_folder / folder / f'{frame}.{ext}'))
            else:
                print(str(src_folder / folder / f'{frame}.{ext}'), "already exists!")


# manual fix for objects labeled incorrectly / ambigiously in scannet scenes
scene_specific_fixes_objectid = {
    "scene0050_02": {
        24: 37,
        26: 37,
        12: 6,
        1: 6,
        16: 9
    },
    "scene0144_01": {
        4: 3,
        13: 3,
        5: 3
    },
    "scene0221_01": {
        15: 7,
        36: 15,
        37: 1,
        38: 1
    },
    "scene0300_01": {
        13: 25,
        14: 25,
        20: 37
    },
    "scene0389_00": {
        19: 37,
        20: 3,
        21: 3,
        28: 37
    },
    "scene0423_02": {
        6: 7
    },
    "scene0616_00": {
        21: 1,
        22: 1,
        24: 1,
        25: 1,
        30: 1,
        31: 1,
    },
    "scene0645_02": {
        5: 7,
        6: 3,
        25: 5,
        41: 5,
        27: 37,
        32: 37,
        34: 37,
        60: 0,
        61: 37
    },
    "scene0693_00": {
        1: 1,
        4: 1,
        6: 3,
        11: 8,
        20: 40,
    }
}
mmdet_export_fixes = {
    "office_020737": {
        "rotation": [math.radians(0.997707), -0.017, -0.065, -0.001],
        "translation": [0, 0, 1],
        "scale": 0.33
    },
    "office_0213meeting": {
        "rotation": None,
        "translation": [0, 0, 0.75],
        "scale": 0.33
    },
    "koenig_0200": {
        "rotation": [math.radians(0.9999), -0.034897, -0.013082, -0.000457],
        "translation": [0, 0, 1.1],
        "scale": 0.25
    }
}


def extract_scan(path_sens_root, path_dest):
    sd = SensorData(str(path_sens_root / f'{path_sens_root.stem}.sens'))
    sd.export_depth_images(str(path_dest / 'depth'))
    sd.export_color_images(str(path_dest / 'color'))
    sd.export_poses(str(path_dest / 'pose'))
    sd.export_intrinsics(str(path_dest / 'intrinsic'))


def extract_labels(path_sens_root, path_dest):
    os.system(f'unzip {str(path_sens_root / f"{path_sens_root.stem}_2d-label-filt.zip")} -d {str(path_dest)}')
    os.system(f'unzip {str(path_sens_root / f"{path_sens_root.stem}_2d-instance-filt.zip")} -d {str(path_dest)}')
    if (path_dest / "instance").exists():
        shutil.rmtree(str(path_dest / "instance"))
    if (path_dest / "semantics").exists():
        shutil.rmtree(str(path_dest / "semantics"))
    os.rename(str(path_dest / "instance-filt"), str(path_dest / "instance"))
    os.rename(str(path_dest / "label-filt"), str(path_dest / "semantics"))


def visualize_mask_folder(path_to_folder, offset=0):
    (path_to_folder.parent / f"visualized_{path_to_folder.stem}").mkdir(exist_ok=True)
    for f in tqdm(list(path_to_folder.iterdir()), desc='visualizing masks'):
        visualize_mask(np.array(Image.open(f)) + offset, path_to_folder.parent / f"visualized_{path_to_folder.stem}" / f.name)


def visualize_confidence_notta(path_to_confidence_file):
    data = torch.load(gzip.open(path_to_confidence_file), map_location='cpu')
    semantics = np.array(Image.open(path_to_confidence_file.parents[1] / "m2f_notta_semantics" / f"{path_to_confidence_file.stem}.png"))
    probability, confidence, confidence_notta = data['probabilities'], data['confidences'], data['confidences_notta']
    confidence_notta[semantics == 0] = 0
    Image.fromarray((cm.get_cmap('gray')(confidence_notta) * 255).astype(np.uint8)).save(f"{path_to_confidence_file.stem}.png")


def visualize_confidence(path_to_confidence_file):
    data = torch.load(gzip.open(path_to_confidence_file), map_location='cpu')
    semantics = np.array(Image.open(path_to_confidence_file.parents[1] / "m2f_semantics" / f"{path_to_confidence_file.stem}.png"))
    probability, confidence, confidence_notta = data['probabilities'], data['confidences'], data['confidences_notta']
    confidence[semantics == 0] = 0
    Image.fromarray((cm.get_cmap('gray')(confidence) * 255).astype(np.uint8)).save(f"{path_to_confidence_file.stem}_tta.png")


def visualize_labels(src_folder):
    visualize_mask_folder(src_folder / "instance")
    visualize_mask_folder(src_folder / "semantics")


def get_scannet_to_nyu_map():
    scannetid_to_nyuid = {int(x.split('\t')[0]): x.split('\t')[4] for x in Path("resources/scannet-labels.combined.tsv").read_text().splitlines()[1:]}
    scannetid_to_nyuid[0] = 0
    scannetid_to_nyuid_arr = np.ones(1280, dtype=np.int32) * 40

    for scid, nyuid in scannetid_to_nyuid.items():
        if nyuid == '':
            nyuid = 40
        else:
            nyuid = int(nyuid)
        scannetid_to_nyuid_arr[scid] = nyuid
    return scannetid_to_nyuid_arr


def scannet_to_nyu(semantics):
    scannetid_to_nyuid_arr = get_scannet_to_nyu_map()
    nyu_semantics = semantics.reshape(-1)
    nyu_semantics = scannetid_to_nyuid_arr[nyu_semantics.tolist()]
    return nyu_semantics.reshape(semantics.shape)


def get_reduce_and_fold_map():
    all_classes = []
    for cllist in [x.strip().split(',') for x in Path("resources/scannet_to_reduced_scannet.csv").read_text().strip().splitlines()]:
        all_classes.append(cllist[0])
    reduce_map = np.zeros(41).astype(np.int)
    for idx, cllist in enumerate([x.strip().split(',') for x in Path("resources/scannet_to_reduced_scannet.csv").read_text().strip().splitlines()]):
        if cllist[1] != '':
            reduce_map[idx + 1] = all_classes.index(cllist[1]) + 1
        else:
            reduce_map[idx + 1] = idx + 1
    fold_map = np.zeros(41).astype(np.int)
    for idx, cllist in enumerate([x.strip().split(',') for x in Path("resources/scannet_reduced_to_coco.csv").read_text().strip().splitlines()]):
        fold_map[all_classes.index(cllist[0]) + 1] = idx + 1
    return reduce_map, fold_map


def fold_scannet_classes(src_folder):
    reduce_map, fold_map = get_reduce_and_fold_map()
    output_folder = src_folder / "rs_semantics"
    output_folder.mkdir(exist_ok=True)
    for f in tqdm((src_folder / "semantics").iterdir(), desc='folding semantics'):
        arr = scannet_to_nyu(np.array(Image.open(f)))
        ins_arr = np.array(Image.open(src_folder / "instance" / f.name))
        if src_folder.stem in scene_specific_fixes_objectid:
            for ob_id in scene_specific_fixes_objectid[src_folder.stem]:
                arr[ins_arr == ob_id] = scene_specific_fixes_objectid[src_folder.stem][ob_id]
        shape = arr.shape
        Image.fromarray(fold_map[reduce_map[arr.flatten()]].reshape(shape).astype(np.int8)).save(output_folder / f.name)


def get_thing_semantics(sc_classes='reduced'):
    thing_semantics = [False]
    for cllist in [x.strip().split(',') for x in Path(f"resources/scannet_{sc_classes}_things.csv").read_text().strip().splitlines()]:
        thing_semantics.append(bool(int(cllist[1])))
    return thing_semantics


def get_classnames(sc_classes='reduced'):
    classnames = ["void"]
    for cllist in [x.strip().split(',') for x in Path(f"resources/scannet_{sc_classes}_things.csv").read_text().strip().splitlines()]:
        classnames.append(cllist[0])
    return classnames


def renumber_instances(src_folder, prefix='rs'):
    all_frame_names = sorted([x.stem for x in (src_folder / f"color").iterdir() if x.name.endswith('.jpg')], key=lambda y: int(y))
    thing_semantics = get_thing_semantics()
    print('len thing_semantics', len(thing_semantics))
    semantics, instances = [], []
    for frame_name in tqdm(all_frame_names, desc='read labels'):
        semantics.append(torch.from_numpy(np.array(Image.open(src_folder / f"{prefix}_semantics" / f"{frame_name}.png"))))
        instances.append(torch.from_numpy(np.array(Image.open(src_folder / f"instance" / f"{frame_name}.png"))))
    semantics = torch.stack(semantics, 0)
    instances = torch.stack(instances, 0)

    instance_semantics_counts = defaultdict(Counter)
    unique_instances = torch.unique(instances)
    for instance in unique_instances:
        usem, uctr = torch.unique(semantics[instances == instance], return_counts=True)
        for usem_idx in range(usem.shape[0]):
            instance_semantics_counts[instance.item()][usem[usem_idx].item()] += uctr[usem_idx].item()
    instance_to_semantic = {}
    for instance in instance_semantics_counts:
        instance_to_semantic[instance] = instance_semantics_counts[instance].most_common(1)[0][0]

    instance_to_remapped_instance = {}
    remapped_instance_to_instance = {0: 0}
    new_instance_id = 1
    for instance in sorted(instance_to_semantic.keys()):
        if thing_semantics[instance_to_semantic[instance]]:
            instance_to_remapped_instance[instance] = new_instance_id
            remapped_instance_to_instance[new_instance_id] = instance
            new_instance_id += 1
        else:
            instance_to_remapped_instance[instance] = 0

    pprint(instance_to_remapped_instance)

    remapped_instances = torch.zeros_like(instances)
    for uinst in unique_instances:
        remapped_instances[instances == uinst.item()] = instance_to_remapped_instance[uinst.item()]

    if (src_folder / 'segmentation_data.pkl').exists():
        export_dict = pickle.load(open(src_folder / 'segmentation_data.pkl', 'rb'))
    else:
        export_dict = {}
    export_dict['num_semantic_classes'] = len(thing_semantics)
    export_dict['fg_classes'] = [i for i, is_thing in enumerate(thing_semantics) if is_thing]
    export_dict['bg_classes'] = [i for i, is_thing in enumerate(thing_semantics) if not is_thing]
    instance_to_semantic[0] = 0
    remapped_instance_to_semantic = {k: instance_to_semantic[remapped_instance_to_instance[k]] for k in range(new_instance_id)}
    export_dict[f'{prefix}_instance_to_semantic'] = remapped_instance_to_semantic

    Path(src_folder / f"{prefix}_instance").mkdir(exist_ok=True)

    # save instances
    for iidx in range(remapped_instances.shape[0]):
        Image.fromarray(remapped_instances[iidx].numpy()).save(src_folder / f"{prefix}_instance" / f"{all_frame_names[iidx]}.png")
    # save bboxes
    pickle.dump(export_dict, open(src_folder / 'segmentation_data.pkl', 'wb'))


def create_inconsistent_instance_map_dataset(src_folder, prefix='rs'):
    all_frame_names = sorted([x.stem for x in (src_folder / "color").iterdir() if x.name.endswith('.jpg')], key=lambda y: int(y))
    sample_indices = list(range(len(all_frame_names)))
    export_dict = pickle.load(open(src_folder / 'segmentation_data.pkl', 'rb'))
    semantics, instances = [], []
    for frame_name in tqdm(all_frame_names, desc='read labels'):
        semantics.append(torch.from_numpy(np.array(Image.open(src_folder / f"{prefix}_semantics" / f"{frame_name}.png"))))
        instances.append(torch.from_numpy(np.array(Image.open(src_folder / f"{prefix}_instance" / f"{frame_name}.png"))))
    semantics = torch.stack(semantics, 0)
    instances = torch.stack(instances, 0)

    instance_to_semantics = export_dict[f'{prefix}_instance_to_semantic']
    fg_classes = export_dict['fg_classes']

    print(instance_to_semantics)
    remapped_instances_inc = instances.clone().long()
    remapped_instances_sem = instances.clone()
    remapped_instance_to_semantics_inc = {}
    new_instance_ctr = 1

    for sidx in tqdm(sorted(list(set(instance_to_semantics.values())))):
        for iidx in range(instances.shape[0]):
            for inst_id in [x for x in sorted(torch.unique(instances[iidx]).tolist()) if x != 0]:
                if instance_to_semantics[inst_id] == sidx:
                    remapped_instances_inc[iidx][instances[iidx] == inst_id] = new_instance_ctr
                    remapped_instance_to_semantics_inc[new_instance_ctr] = sidx
                    new_instance_ctr += 1

    print(remapped_instances_inc.max())

    for i in range(len(fg_classes)):
        remapped_instances_sem[semantics == fg_classes[i]] = i + 1

    Path(src_folder / f"{prefix}_instance_inc").mkdir(exist_ok=True)
    Path(src_folder / f"{prefix}_instance_sem").mkdir(exist_ok=True)

    for iidx in tqdm(range(remapped_instances_inc.shape[0])):
        sample_index = sample_indices[iidx]
        Image.fromarray(remapped_instances_inc[iidx].numpy().astype(np.uint16)).save(src_folder / f"{prefix}_instance_inc" / f"{all_frame_names[sample_index]}.png")
        Image.fromarray(remapped_instances_sem[iidx].numpy()).save(src_folder / f"{prefix}_instance_sem" / f"{all_frame_names[sample_index]}.png")

    export_dict[f'{prefix}_instance_to_semantic_inc'] = remapped_instance_to_semantics_inc
    pickle.dump(export_dict, open(src_folder / 'segmentation_data.pkl', 'wb'))


def convert_from_mask_to_semantics_and_instances(original_mask, segments, coco_to_scannet, is_thing, instance_ctr, instance_to_semantic):
    id_to_class = torch.zeros(1024).int()
    instance_mask = torch.zeros_like(original_mask)
    invalid_mask = original_mask == 0
    for s in segments:
        if s['category_name'] in coco_to_scannet:
            id_to_class[s['id']] = coco_to_scannet[s['category_name']]
            if is_thing[coco_to_scannet[s['category_name']]]:
                instance_mask[original_mask == s['id']] = instance_ctr
                instance_to_semantic[instance_ctr] = coco_to_scannet[s['category_name']]
                instance_ctr += 1
    return id_to_class[original_mask.flatten().numpy().tolist()].reshape(original_mask.shape), instance_mask, invalid_mask, instance_ctr, instance_to_semantic


def convert_from_mask_to_semantics_and_instances_no_remap(original_mask, segments, _coco_to_scannet, is_thing, instance_ctr, instance_to_semantic):
    id_to_class = torch.zeros(1024).int()
    instance_mask = torch.zeros_like(original_mask)
    invalid_mask = original_mask == 0
    for s in segments:
        id_to_class[s['id']] = s['category_id']
        if is_thing[s['category_id']]:
            instance_mask[original_mask == s['id']] = instance_ctr
            instance_to_semantic[instance_ctr] = s['category_id']
            instance_ctr += 1
    return id_to_class[original_mask.flatten().numpy().tolist()].reshape(original_mask.shape), instance_mask, invalid_mask, instance_ctr, instance_to_semantic


def map_panoptic_coco(src_folder, sc_classes='reduced', undistort=False):
    coco_to_scannet = {}
    thing_semantics = get_thing_semantics(sc_classes)
    for cidx, cllist in enumerate([x.strip().split(',') for x in Path(f"resources/scannet_{sc_classes}_to_coco.csv").read_text().strip().splitlines()]):
        for c in cllist[1:]:
            coco_to_scannet[c.split('/')[1]] = cidx + 1
    instance_ctr = 1
    instance_to_semantic = {}
    instance_ctr_notta = 1
    segment_ctr = 1
    instance_to_semantic_notta = {}
    (src_folder / "m2f_instance").mkdir(exist_ok=True)
    (src_folder / "m2f_semantics").mkdir(exist_ok=True)
    (src_folder / "m2f_notta_instance").mkdir(exist_ok=True)
    (src_folder / "m2f_notta_semantics").mkdir(exist_ok=True)
    (src_folder / "m2f_feats").mkdir(exist_ok=True)
    (src_folder / "m2f_probabilities").mkdir(exist_ok=True)
    (src_folder / "m2f_invalid").mkdir(exist_ok=True)
    (src_folder / "m2f_segments").mkdir(exist_ok=True)

    if undistort:
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
            undistort = False

    for idx, fpath in enumerate(tqdm(sorted(list((src_folder / "color").iterdir()), key=lambda x: x.stem), desc='map labels')):
        data = torch.load(gzip.open(src_folder / "panoptic" / f'{fpath.stem}.ptz'), map_location='cpu')
        probability, confidence, confidence_notta = data['probabilities'], data['confidences'], data['confidences_notta']

        if undistort:
            probability_numpy = probability.cpu().numpy()
            probability_undistorted = np.zeros_like(probability_numpy)
            for cidx in range(probability_numpy.shape[-1]):
                probability_undistorted[:, :, cidx] = cv2.remap(probability_numpy[:, :, cidx], mapx, mapy, cv2.INTER_CUBIC)
            confidence_undistorted = cv2.remap(confidence.cpu().numpy(), mapx, mapy, cv2.INTER_CUBIC)
            confidence_notta_undistorted = cv2.remap(confidence_notta.cpu().numpy(), mapx, mapy, cv2.INTER_CUBIC)
            probability, confidence, confidence_notta = torch.clip(torch.from_numpy(probability_undistorted), 0, 1), torch.clip(torch.from_numpy(confidence_undistorted), 0, 1), torch.clip(torch.from_numpy(confidence_notta_undistorted), 0, 1)

        semantic, instance, invalid_mask, instance_ctr, instance_to_semantic = convert_from_mask_to_semantics_and_instances_no_remap(data['mask'], data['segments'], coco_to_scannet, thing_semantics, instance_ctr, instance_to_semantic)
        semantic_notta, instance_notta, _, instance_ctr_notta, instance_to_semantic_notta = convert_from_mask_to_semantics_and_instances_no_remap(data['mask_notta'], data['segments_notta'], coco_to_scannet, thing_semantics,
                                                                                                                                                  instance_ctr_notta, instance_to_semantic_notta)
        segment_mask = torch.zeros_like(data['mask'])
        for s in data['segments']:
            segment_mask[data['mask'] == s['id']] = segment_ctr
            segment_ctr += 1
        Image.fromarray(segment_mask.numpy().astype(np.uint16)).save(src_folder / "m2f_segments" / f"{fpath.stem}.png")
        Image.fromarray(semantic.numpy().astype(np.uint16)).save(src_folder / "m2f_semantics" / f"{fpath.stem}.png")
        Image.fromarray(instance.numpy()).save(src_folder / "m2f_instance" / f"{fpath.stem}.png")
        Image.fromarray(semantic_notta.numpy().astype(np.uint16)).save(src_folder / "m2f_notta_semantics" / f"{fpath.stem}.png")
        Image.fromarray(instance_notta.numpy()).save(src_folder / "m2f_notta_instance" / f"{fpath.stem}.png")
        Image.fromarray(invalid_mask.numpy().astype(np.uint8) * 255).save(src_folder / "m2f_invalid" / f"{fpath.stem}.png")
        # interpolated_p = torch.nn.functional.interpolate(torch.cat([probability.permute((2, 0, 1)), confidence.unsqueeze(0), confidence_notta.unsqueeze(0)], 0).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        # probability, confidence, confidence_notta = interpolated_p[:-2, :, :].permute((1, 2, 0)).cpu(), interpolated_p[-2, :, :].cpu(), interpolated_p[-1, :, :].cpu()
        np.savez_compressed(src_folder / "m2f_probabilities" / f"{fpath.stem}.npz", probability=probability.float().numpy(), confidence=confidence.float().numpy(), confidence_notta=confidence_notta.float().numpy())
        
        if undistort:
            to_undistort = [
                src_folder / "m2f_segments" / f"{fpath.stem}.png",
                src_folder / "m2f_semantics" / f"{fpath.stem}.png",
                src_folder / "m2f_instance" / f"{fpath.stem}.png",
                src_folder / "m2f_notta_semantics" / f"{fpath.stem}.png",
                src_folder / "m2f_notta_instance" / f"{fpath.stem}.png",
                src_folder / "m2f_invalid" / f"{fpath.stem}.png"
            ]
            for img_p in to_undistort:
                img = np.array(Image.open(img_p))
                img = cv2.remap(img, mapx, mapy, cv2.INTER_NEAREST)
                img[distorted_mask > 0] = 0
                img = img[roi[1]: roi[3] + 1, roi[0]: roi[2] + 1]
                Image.fromarray(img).save(img_p)

        # feats = data['feats']
        # np.savez_compressed(src_folder / "m2f_feats" / f"{fpath.stem}.npz", feats=feats.float().numpy())

    export_dict = pickle.load(open(src_folder / 'segmentation_data.pkl', 'rb'))
    export_dict[f'm2f_instance_to_semantic'] = instance_to_semantic
    export_dict[f'm2f_notta_instance_to_semantic'] = instance_to_semantic
    pprint(instance_to_semantic)
    pickle.dump(export_dict, open(src_folder / 'segmentation_data.pkl', 'wb'))


def map_gt_bboxes(path_sens_root, src_folder):
    reduce_map, fold_map = get_reduce_and_fold_map()
    thing_semantics = get_thing_semantics()
    distinct_colors = DistinctColors()
    bboxes = {}
    valid_boxid = 0
    (src_folder / "visualized_gtboxes").mkdir(exist_ok=True)
    pkl_segmentation_data = pickle.load(open(src_folder / f'segmentation_data.pkl', 'rb'))
    bbox_annot = np.load(path_sens_root / f"{path_sens_root.stem}_bbox.npy")
    for bbox_idx in range(bbox_annot.shape[0]):
        position = bbox_annot[bbox_idx][0:3]
        orientation = np.eye(3)
        extent = bbox_annot[bbox_idx][3:6]
        instance_id = int(bbox_annot[bbox_idx][7]) + 1
        if src_folder.stem in scene_specific_fixes_objectid:
            if instance_id in scene_specific_fixes_objectid[src_folder.stem]:
                bbox_annot[bbox_idx][6] = scene_specific_fixes_objectid[src_folder.stem][instance_id]
        label = fold_map[reduce_map[int(bbox_annot[bbox_idx][6])]]
        if thing_semantics[label]:
            bboxes[valid_boxid] = {
                'position': position,
                'orientation': orientation,
                'extent': extent,
                'class': label
            }
            create_box(position, extent, orientation, distinct_colors.get_color_fast_numpy(label)).export(src_folder / "visualized_gtboxes" / f"{label}_{valid_boxid}.obj")
            valid_boxid += 1
    pkl_segmentation_data['gt_bboxes'] = bboxes
    pickle.dump(pkl_segmentation_data, open(src_folder / f'segmentation_data.pkl', 'wb'))


def map_imvoxnet_boxes(path_bboxes, src_folder, class_set="reduced"):
    mmdet_to_scannet_reduced = {}
    for cidx, cllist in enumerate([x.strip().split(',') for x in Path(f"resources/scannet_mmdet_to_scannet_{class_set}.csv").read_text().strip().splitlines()]):
        mmdet_to_scannet_reduced[cllist[0]] = cllist[1]
    classes = [""]
    for idx, cllist in enumerate([x.strip().split(',') for x in Path(f"resources/scannet_{class_set}_to_coco.csv").read_text().strip().splitlines()]):
        classes.append(cllist[0])
    thing_semantics = get_thing_semantics()
    distinct_colors = DistinctColors()
    bboxes = {}
    valid_boxid = 0
    (src_folder / "visualized_mmdetboxes").mkdir(exist_ok=True)
    pkl_segmentation_data = pickle.load(open(src_folder / f'segmentation_data.pkl', 'rb'))
    bbox_annot = json.loads(Path(path_bboxes).read_text())
    for bbox in bbox_annot:
        corners = np.array(bbox['corners'])
        if src_folder.stem in mmdet_export_fixes:
            rotation_fix = np.eye(4)
            axangle = mmdet_export_fixes[src_folder.stem]["rotation"]
            if axangle is not None:
                rotation_fix[:3, :3] = axangle2mat(axangle[1:4], axangle[0])
            translation_fix = hmg(np.eye(3))
            translation_fix[:3, 3] = np.array(mmdet_export_fixes[src_folder.stem]["translation"])
            scale_fix = hmg(np.eye(3) * mmdet_export_fixes[src_folder.stem]["scale"])
            corners = dot(np.linalg.inv(translation_fix @ scale_fix @ rotation_fix), corners)
        cmin = np.min(corners, axis=0)
        cmax = np.max(corners, axis=0)
        position = (cmax + cmin) / 2
        orientation = np.eye(3)
        label = classes.index(mmdet_to_scannet_reduced[bbox['label']].lower())
        extent = cmax - cmin
        if thing_semantics[label]:
            bboxes[valid_boxid] = {
                'position': position,
                'orientation': orientation,
                'extent': extent,
                'class': label
            }
            create_box(position, extent, orientation, distinct_colors.get_color_fast_numpy(label)).export(src_folder / "visualized_mmdetboxes" / f"{label}_{valid_boxid}.obj")
            valid_boxid += 1
    pkl_segmentation_data['mmdet_bboxes'] = bboxes
    pickle.dump(pkl_segmentation_data, open(src_folder / f'segmentation_data.pkl', 'wb'))


def read_and_resize_labels(path, size):
    image = Image.open(path)
    return np.array(image.resize(size, Image.NEAREST))


def calculate_iou_folders_image_wise(path_pred, path_target, image_size, pred_offset=0):
    num_semantic_classes = 1 + len(Path("resources/scannet_reduced_to_coco.csv").read_text().strip().splitlines())
    iou_avg = 0
    val_set = json.loads(Path(path_target.parent / "splits.json").read_text())['test']
    val_paths = [y for y in sorted(list(path_pred.iterdir()), key=lambda x: int(x.stem)) if y.stem in val_set]
    faulty_gt_classes = [0]
    for p in tqdm(val_paths):
        img_pred = read_and_resize_labels(p, image_size) + pred_offset
        img_target = read_and_resize_labels(path_target / p.name, image_size)
        valid_mask = ~np.isin(img_target, faulty_gt_classes)
        train_cm = ConfusionMatrix(num_classes=num_semantic_classes, ignore_class=[])
        iou = train_cm.add_batch(img_pred[valid_mask], img_target[valid_mask], return_miou=True)
        iou_avg += iou
    iou_avg /= len(val_paths)
    return iou_avg


def calculate_iou_folders(path_pred, path_target, image_size, pred_offset=0):
    num_semantic_classes = 1 + len(Path("resources/scannet_reduced_to_coco.csv").read_text().strip().splitlines())
    val_set = json.loads(Path(path_target.parent / "splits.json").read_text())['test']
    val_paths = [y for y in sorted(list(path_pred.iterdir()), key=lambda x: int(x.stem)) if y.stem in val_set]
    faulty_gt_classes = [0]
    train_cm = ConfusionMatrix(num_classes=num_semantic_classes, ignore_class=[])
    for p in tqdm(val_paths):
        img_pred = read_and_resize_labels(p, image_size) + pred_offset
        img_target = read_and_resize_labels(path_target / p.name, image_size)
        valid_mask = ~np.isin(img_target, faulty_gt_classes)
        train_cm.add_batch(img_pred[valid_mask], img_target[valid_mask], return_miou=False)
    return train_cm.get_miou()


def calculate_panoptic_quality_folders_image_wise(path_pred_sem, path_pred_inst, path_target_sem, path_target_inst, image_size):
    is_thing = get_thing_semantics()
    val_set = json.loads(Path(path_target_sem.parent / "splits.json").read_text())['test']
    faulty_gt = [0]
    things = set([i for i in range(len(is_thing)) if is_thing[i]])
    stuff = set([i for i in range(len(is_thing)) if not is_thing[i]])
    pq_avg, sq_avg, rq_avg = 0, 0, 0
    val_paths = [y for y in sorted(list(path_pred_sem.iterdir()), key=lambda x: int(x.stem)) if y.stem in val_set]
    for p in tqdm(val_paths):
        img_target_sem = read_and_resize_labels((path_target_sem / p.name), image_size)
        valid_mask = ~np.isin(img_target_sem, faulty_gt)
        img_pred_sem = torch.from_numpy(read_and_resize_labels(p, image_size)[valid_mask]).unsqueeze(-1)
        img_target_sem = torch.from_numpy(img_target_sem[valid_mask]).unsqueeze(-1)
        img_pred_inst = torch.from_numpy(read_and_resize_labels((path_pred_inst / p.name), image_size)[valid_mask]).unsqueeze(-1)
        img_target_inst = torch.from_numpy(read_and_resize_labels((path_target_inst / p.name), image_size)[valid_mask]).unsqueeze(-1)
        pred = torch.cat([img_pred_sem, img_pred_inst], dim=1).cuda()
        target = torch.cat([img_target_sem, img_target_inst], dim=1).cuda()
        pq, sq, rq = panoptic_quality(pred, target, things, stuff, allow_unknown_preds_category=True)
        pq_avg += pq.item()
        sq_avg += sq.item()
        rq_avg += rq.item()
    pq_avg /= len(val_paths)
    sq_avg /= len(val_paths)
    rq_avg /= len(val_paths)
    return pq_avg, sq_avg, rq_avg


def calculate_panoptic_quality_folders(path_pred_sem, path_pred_inst, path_target_sem, path_target_inst, image_size):
    is_thing = get_thing_semantics()
    val_set = json.loads(Path(path_target_sem.parent / "splits.json").read_text())['test']
    faulty_gt = [0]
    things = set([i for i in range(len(is_thing)) if is_thing[i]])
    stuff = set([i for i in range(len(is_thing)) if not is_thing[i]])
    val_paths = [y for y in sorted(list(path_pred_sem.iterdir()), key=lambda x: int(x.stem)) if y.stem in val_set]
    pred, target = [], []
    for p in tqdm(val_paths):
        img_target_sem = read_and_resize_labels((path_target_sem / p.name), image_size)
        valid_mask = ~np.isin(img_target_sem, faulty_gt)
        img_pred_sem = torch.from_numpy(read_and_resize_labels(p, image_size)[valid_mask]).unsqueeze(-1)
        img_target_sem = torch.from_numpy(img_target_sem[valid_mask]).unsqueeze(-1)
        img_pred_inst = torch.from_numpy(read_and_resize_labels((path_pred_inst / p.name), image_size)[valid_mask]).unsqueeze(-1)
        img_target_inst = torch.from_numpy(read_and_resize_labels((path_target_inst / p.name), image_size)[valid_mask]).unsqueeze(-1)
        pred_ = torch.cat([img_pred_sem, img_pred_inst], dim=1).reshape(-1, 2)
        target_ = torch.cat([img_target_sem, img_target_inst], dim=1).reshape(-1, 2)
        pred.append(pred_)
        target.append(target_)
    pq, sq, rq = panoptic_quality(torch.cat(pred, dim=0).cuda(), torch.cat(target, dim=0).cuda(), things, stuff, allow_unknown_preds_category=True)
    return pq.item(), sq.item(), rq.item()


def calculate_panoptic_quality_per_frame_folders(path_pred_sem, path_pred_inst, path_target_sem, path_target_inst, image_size):
    is_thing = get_thing_semantics()
    val_set = json.loads(Path(path_target_sem.parent / "splits.json").read_text())['test']
    faulty_gt = [0]
    things = set([i for i in range(len(is_thing)) if is_thing[i]])
    stuff = set([i for i in range(len(is_thing)) if not is_thing[i]])
    val_paths = [y for y in sorted(list(path_pred_sem.iterdir()), key=lambda x: int(x.stem)) if y.stem in val_set]
    things_, stuff_, iou_sum_, true_positives_, false_positives_, false_negatives_ = set(), set(), [], [], [], []
    for p in tqdm(val_paths):
        img_target_sem = read_and_resize_labels((path_target_sem / p.name), image_size)
        valid_mask = ~np.isin(img_target_sem, faulty_gt)
        img_pred_sem = torch.from_numpy(read_and_resize_labels(p, image_size)[valid_mask]).unsqueeze(-1)
        img_target_sem = torch.from_numpy(img_target_sem[valid_mask]).unsqueeze(-1)
        img_pred_inst = torch.from_numpy(read_and_resize_labels((path_pred_inst / p.name), image_size)[valid_mask]).unsqueeze(-1)
        img_target_inst = torch.from_numpy(read_and_resize_labels((path_target_inst / p.name), image_size)[valid_mask]).unsqueeze(-1)
        pred_ = torch.cat([img_pred_sem, img_pred_inst], dim=1).reshape(-1, 2)
        target_ = torch.cat([img_target_sem, img_target_inst], dim=1).reshape(-1, 2)
        _things, _stuff, _iou_sum, _true_positives, _false_positives, _false_negatives = panoptic_quality_match(pred_, target_, things, stuff, True)
        things_.union(_things)
        stuff_.union(_stuff)
        iou_sum_.append(_iou_sum)
        true_positives_.append(_true_positives)
        false_positives_.append(_false_positives)
        false_negatives_.append(_false_negatives)
    results = _panoptic_quality_compute(things_, stuff_, torch.cat(iou_sum_, 0), torch.cat(true_positives_, 0), torch.cat(false_positives_, 0), torch.cat(false_negatives_, 0))
    return results["all"]["pq"].item(), results["all"]["sq"].item(), results["all"]["rq"].item()


def create_validation_set(src_folder, fraction):
    all_frames = [x.stem for x in sorted(list((src_folder / "color").iterdir()), key=lambda x: int(x.stem))]
    selected_val = [all_frames[i] for i in range(0, len(all_frames), int(1 / fraction))]
    selected_train = [x for x in all_frames if x not in selected_val]
    print(len(selected_train), len(selected_val))
    Path(src_folder / "splits.json").write_text(json.dumps({
        'train': selected_train,
        'test': selected_val
    }))


def create_mask2former_split_data(src_folder):
    all_frame_names = sorted([x.stem for x in (src_folder / f"color").iterdir() if x.name.endswith('.jpg')], key=lambda y: int(y) if y.isnumeric() else y)
    thing_semantics = get_thing_semantics()
    print('len thing_semantics', len(thing_semantics))
    semantics = []
    for frame_name in tqdm(all_frame_names, desc='read labels'):
        semantics.append(torch.from_numpy(np.array(Image.open(src_folder / f"m2f_semantics" / f"{frame_name}.png"))))

    semantics = torch.stack(semantics, 0)
    export_dict = pickle.load(open(src_folder / 'segmentation_data.pkl', 'rb'))
    instance_to_semantics = export_dict[f'm2f_instance_to_semantic']
    fg_classes = export_dict['fg_classes']

    print(instance_to_semantics)
    remapped_instances_sem = torch.zeros_like(semantics)

    sem_instance_to_semantics = {0: 0}
    for i in range(len(fg_classes)):
        remapped_instances_sem[semantics == fg_classes[i]] = i + 1
        sem_instance_to_semantics[i + 1] = fg_classes[i]

    Path(src_folder / f"m2f_instance_sem").mkdir(exist_ok=True)

    for iidx in tqdm(range(semantics.shape[0])):
        Image.fromarray(remapped_instances_sem[iidx].numpy()).save(src_folder / f"m2f_instance_sem" / f"{all_frame_names[iidx]}.png")

    export_dict['m2f_sem_instance_to_semantics'] = sem_instance_to_semantics
    pickle.dump(export_dict, open(src_folder / f'segmentation_data.pkl', 'wb'))


def create_m2f_used_instances(src_folder):
    export_dict = pickle.load(open(src_folder / 'segmentation_data.pkl', 'rb'))
    instance_to_semantics = export_dict[f'm2f_sem_instance_to_semantics']
    print(instance_to_semantics)
    all_frame_names = sorted([x.stem for x in (src_folder / f"color").iterdir() if x.name.endswith('.jpg')], key=lambda y: int(y) if y.isnumeric() else y)
    frame_counts = {k: 0 for k in instance_to_semantics.keys()}
    dims = Image.open(src_folder / f"m2f_instance_sem" / f"{all_frame_names[0]}.png").size
    for frame_name in tqdm(all_frame_names, desc='read labels'):
        uinsts, ucounts = torch.from_numpy(np.array(Image.open(src_folder / f"m2f_instance_sem" / f"{frame_name}.png"))).unique(return_counts=True)
        for iidx in range(len(uinsts)):
            percinst = ucounts[iidx] / (dims[0] * dims[1])
            if percinst > 0.005:
                frame_counts[uinsts[iidx].item()] += 1
    is_valid_instance = {}
    for k in frame_counts:
        is_valid_instance[k] = True if frame_counts[k] > len(all_frame_names) * 0.01 else False
    print(is_valid_instance)
    export_dict['m2f_sem_valid_instance'] = is_valid_instance
    pickle.dump(export_dict, open(src_folder / f'segmentation_data.pkl', 'wb'))


def create_instances_for_dmnerf(src_folder, correspondences, class_set='reduced'):
    suffix_o = "_no_correspondences" if not correspondences else ""
    suffix_i = "_correspondences" if correspondences else ""
    color_folder = src_folder / "color"
    semantics_folder = src_folder / "m2f_notta_semantics"
    instance_folder = src_folder / f"m2f_notta_instance{suffix_i}"
    output_folder = src_folder / f"m2f_notta_dmnerf{suffix_o}"
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(exist_ok=True)
    is_thing_class = get_thing_semantics(sc_classes=class_set)
    stuff_classes = [i for i in range(len(is_thing_class)) if not is_thing_class[i]]
    instance_to_semantics = {}
    for f in tqdm(list(color_folder.iterdir()), desc='creating new mask'):
        semantics = np.array(Image.open(semantics_folder / f"{f.stem}.png"))
        instance = np.array(Image.open(instance_folder / f"{f.stem}.png"))
        classes = np.unique(semantics)
        new_instance = np.zeros_like(instance)
        for c in classes:
            if c in stuff_classes:
                assigned_index = stuff_classes.index(c)
                new_instance[semantics == c] = assigned_index
                instance_to_semantics[assigned_index] = int(c)
            else:
                uniques = np.unique(instance[semantics == c])
                for u in uniques:
                    if u != 0:
                        assigned_index = len(stuff_classes) + u
                        new_instance[instance == u] = assigned_index
                        instance_to_semantics[assigned_index] = int(c)
        Image.fromarray(new_instance).save(output_folder / f"{f.stem}.png")
    pickle.dump(instance_to_semantics, open(src_folder / f"dmnerf_i2s{suffix_o}.pkl", "wb"))


def from_ours_to_replica_traj_w_c(src_folder):
    poses = sorted(list((src_folder / "pose").iterdir()), key=lambda x: int(x.stem) if x.stem.isnumeric() else x.stem)
    traj_w_c_string = ""
    for pose_file in poses:
        RT = np.array([[float(y.strip()) for y in x.strip().split()] for x in Path(pose_file).read_text().splitlines() if x != ''])
        traj_w_c_string += f"""{RT[0, 0]} {RT[0, 1]} {RT[0, 2]} {RT[0, 3]} {RT[1, 0]} {RT[1, 1]} {RT[1, 2]} {RT[1, 3]} {RT[2, 0]} {RT[2, 1]} {RT[2, 2]} {RT[2, 3]} 0.00 0.00 0.00 1.00\n"""
    (src_folder / "traj_w_c.txt").write_text(traj_w_c_string)


def from_trajectory_to_replica_traj_blend(src_folder):
    traj_w_c_string = ""
    with open(src_folder / "trajectories" / f"trajectory_blender.pkl", "rb") as fptr:
        trajectories = pickle.load(fptr)
    for i in range(len(trajectories)):
        RT = trajectories[i]
        traj_w_c_string += f"""{RT[0, 0]} {RT[0, 1]} {RT[0, 2]} {RT[0, 3]} {RT[1, 0]} {RT[1, 1]} {RT[1, 2]} {RT[1, 3]} {RT[2, 0]} {RT[2, 1]} {RT[2, 2]} {RT[2, 3]} 0.00 0.00 0.00 1.00\n"""
    (src_folder / "traj_blender.txt").write_text(traj_w_c_string)


def debug_dump_instances_for_scene(path):
    instance = np.array(Image.open(path))
    u, c = np.unique(instance, return_counts=True)
    for uin in u:
        visualize_mask((instance == uin).astype(int), f"inst_{uin}.png")


def export_all_for_semantic_nerf(src_folder):
    base_dir = src_folder.parent
    all_scenes = [x for x in base_dir.iterdir() if x.name.startswith("scene")]
    for scene in tqdm(all_scenes):
        out_dir = base_dir / "raw" / "from_semantic_nerf" / scene.name / "Sequence_1"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        # copy color -> rgb
        # remake splits
        splits = json.loads((base_dir / scene.name / "splits.json").read_text())
        for split in ["train", "val"]:
            splits[split] = [f"{int(x):04d}" for x in splits[split]]
        Path(out_dir / "splits.json").write_text(json.dumps(splits))
        # copy intrinsics
        shutil.copyfile(base_dir / scene.name / "intrinsic" / "intrinsic_color.txt", out_dir / "intrinsic_color.txt")
        # make trajectory and copy
        from_ours_to_replica_traj_w_c(base_dir / scene.name)
        shutil.copyfile(base_dir / scene.name / "traj_w_c.txt", out_dir / "traj_w_c.txt")
        (out_dir / "rgb").mkdir()
        for f in (base_dir / scene.name / "color").iterdir():
            shutil.copyfile(f, out_dir / "rgb" / f"{int(f.stem):04d}.jpg")
        # copy depth -> depth
        if not (out_dir / "depth").exists():
            shutil.copytree(base_dir / scene.name / "depth", out_dir / "depth")


def export_all_for_dmnerf(src_folder):
    base_dir = src_folder.parent
    all_scenes = [x for x in base_dir.iterdir() if x.name.startswith("scene")]
    for scene in tqdm(all_scenes):
        dm_nerf_path = Path("/cluster/gimli/ysiddiqui/dm-nerf-data/scannet") / scene.name
        out_dir = dm_nerf_path
        if not out_dir.exists():
            shutil.copytree(base_dir / "raw" / "from_semantic_nerf" / scene.name / "Sequence_1", out_dir)
        create_instances_for_dmnerf(base_dir / scene.name, correspondences=False)
        suffix = "_no_correspondences"
        output_folder = dm_nerf_path / f"semantic_instance_m2f{suffix}"
        output_folder.mkdir(exist_ok=True)
        input_folder = base_dir / scene.name / f"m2f_notta_dmnerf{suffix}"
        input_names = sorted(list(input_folder.iterdir()), key=lambda x: int(x.stem))
        output_names = [f"semantic_instance_{int(x.stem)}" for x in input_names]
        for idx in range(len(input_names)):
            shutil.copyfile(input_names[idx], output_folder / f"{output_names[idx]}.png")


def render_mesh(src_sens_path):
    import trimesh
    import pyrender

    def create_groups():
        seg_file = src_sens_path / f"{sens_root.stem}_vh_clean.segs.json"
        seg_indices = np.array(json.loads(Path(seg_file).read_text())["segIndices"])
        face_seg_ids = np.concatenate([seg_indices[scannet_mesh.faces[:, 0:1]], seg_indices[scannet_mesh.faces[:, 1:2]], seg_indices[scannet_mesh.faces[:, 2:3]]], axis=-1)
        face_seg_ids = stats.mode(face_seg_ids, axis=1).mode[:, 0]
        vertex_reseg = np.zeros_like(seg_indices)
        vertex_reseg[scannet_mesh.faces[:, 0]] = face_seg_ids
        vertex_reseg[scannet_mesh.faces[:, 1]] = face_seg_ids
        vertex_reseg[scannet_mesh.faces[:, 2]] = face_seg_ids
        # colors = (distinct_colors.get_color_fast_numpy(vertex_reseg) * 255).astype(np.uint8)
        # trimesh.Trimesh(vertices=scannet_mesh.vertices, faces=scannet_mesh.faces, vertex_colors=colors).export("test_seg.obj")
        colors = np.concatenate([x[:, np.newaxis] for x in [(vertex_reseg // 256 ** 2) % 256, (vertex_reseg // 256) % 256, vertex_reseg % 256]], axis=-1)
        return trimesh.Trimesh(vertices=scannet_mesh.vertices, faces=scannet_mesh.faces, vertex_colors=colors)

    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    src_folder = Path("data/scannet/", sens_root.stem)
    with open(src_folder / "trajectories" / f"trajectory_blender.pkl", "rb") as fptr:
        trajectories = pickle.load(fptr)
    scannet_mesh = trimesh.load(src_sens_path / "scene0050_02_vh_clean.ply", process=False)
    mesh = pyrender.Mesh.from_trimesh(create_groups())
    scene = pyrender.Scene()
    scene.add(mesh)
    for i, pose in enumerate(tqdm(trajectories)):
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=640 / 480)
        camera_pose = pose @ flip_mat
        camera_node = scene.add(camera, pose=camera_pose)
        r = pyrender.OffscreenRenderer(640, 480)
        color, depth = r.render(scene, pyrender.constants.RenderFlags.FLAT)
        Image.fromarray(color).save(f"runs/scannet_render/segments/{i:04d}.png")
        scene.remove_node(camera_node)


def map_rendered_mesh(src_sens_path):
    import inflect
    inflect_engine = inflect.engine()
    root = Path("runs/scannet_render")
    out_sem = root / "semantics"
    out_ins = root / "instance"
    seg_file = src_sens_path / f"{sens_root.stem}.aggregation.json"
    seg_groups = json.loads(Path(seg_file).read_text())["segGroups"]
    segment_to_label = np.zeros(2 ** 24).astype(np.int32)
    segment_to_id = np.zeros(2 ** 24).astype(np.int32)
    distinct_colors = DistinctColors()
    scannetlabel_to_nyuid = {x.split('\t')[1]: x.split('\t')[4] for x in Path("resources/scannet-labels.combined.tsv").read_text().splitlines()[1:]}
    scannetlabel_to_nyuid['object'] = 40
    for group in seg_groups:
        label = group['label']
        if inflect_engine.singular_noun(group['label']):
            label = inflect_engine.singular_noun(group['label'])
        segment_to_label[group["segments"]] = scannetlabel_to_nyuid[label]
        segment_to_id[group["segments"]] = group['id'] + 1
    bg_classes = [i for i, x in enumerate(get_thing_semantics("reduced")) if not x]
    for item in tqdm(sorted(list((root / "segments").iterdir()), key=lambda x: int(x.stem))):
        segment = np.array(Image.open(item))
        segment = segment[:, :, 0] * 256 ** 2 + segment[:, :, 1] * 256 + segment[:, :, 2]
        ids = segment_to_id[segment]
        segment = segment_to_label[segment]
        ids[segment == 0] = -1
        segment[segment == 0] = -1
        segment[segment == (256 * 256 * 255 + 256 * 255 + 255)] = 0

        for i in range(1):
            arr_t, arr_r, arr_b, arr_l = segment[1:, :], segment[:, 1:], segment[:-1, :], segment[:, :-1]
            arr_t_1, arr_r_1, arr_b_1, arr_l_1 = segment[2:, :], segment[:, 2:], segment[:-2, :], segment[:, :-2]

            arr_t = np.concatenate([arr_t, segment[-1, :][np.newaxis, :]], axis=0)
            arr_r = np.concatenate([arr_r, segment[:, -1][:, np.newaxis]], axis=1)
            arr_b = np.concatenate([segment[0, :][np.newaxis, :], arr_b], axis=0)
            arr_l = np.concatenate([segment[:, 0][:, np.newaxis], arr_l], axis=1)

            arr_t_1 = np.concatenate([arr_t_1, segment[-2, :][np.newaxis, :], segment[-1, :][np.newaxis, :]], axis=0)
            arr_r_1 = np.concatenate([arr_r_1, segment[:, -2][:, np.newaxis], segment[:, -1][:, np.newaxis]], axis=1)
            arr_b_1 = np.concatenate([segment[0, :][np.newaxis, :], segment[1, :][np.newaxis, :], arr_b_1], axis=0)
            arr_l_1 = np.concatenate([segment[:, 0][:, np.newaxis], segment[:, 1][:, np.newaxis], arr_l_1], axis=1)

            segment[np.logical_and(segment == -1, arr_t != -1)] = arr_t[np.logical_and(segment == -1, arr_t != -1)]
            segment[np.logical_and(segment == -1, arr_r != -1)] = arr_r[np.logical_and(segment == -1, arr_r != -1)]
            segment[np.logical_and(segment == -1, arr_b != -1)] = arr_b[np.logical_and(segment == -1, arr_b != -1)]
            segment[np.logical_and(segment == -1, arr_l != -1)] = arr_l[np.logical_and(segment == -1, arr_l != -1)]

            segment[np.logical_and(segment == -1, arr_t_1 != -1)] = arr_t_1[np.logical_and(segment == -1, arr_t_1 != -1)]
            segment[np.logical_and(segment == -1, arr_r_1 != -1)] = arr_r_1[np.logical_and(segment == -1, arr_r_1 != -1)]
            segment[np.logical_and(segment == -1, arr_b_1 != -1)] = arr_b_1[np.logical_and(segment == -1, arr_b_1 != -1)]
            segment[np.logical_and(segment == -1, arr_l_1 != -1)] = arr_l_1[np.logical_and(segment == -1, arr_l_1 != -1)]

            arr_t, arr_r, arr_b, arr_l = ids[1:, :], ids[:, 1:], ids[:-1, :], ids[:, :-1]
            arr_t_1, arr_r_1, arr_b_1, arr_l_1 = ids[2:, :], ids[:, 2:], ids[:-2, :], ids[:, :-2]

            arr_t = np.concatenate([arr_t, ids[-1, :][np.newaxis, :]], axis=0)
            arr_r = np.concatenate([arr_r, ids[:, -1][:, np.newaxis]], axis=1)
            arr_b = np.concatenate([ids[0, :][np.newaxis, :], arr_b], axis=0)
            arr_l = np.concatenate([ids[:, 0][:, np.newaxis], arr_l], axis=1)

            arr_t_1 = np.concatenate([arr_t_1, ids[-2, :][np.newaxis, :], ids[-1, :][np.newaxis, :]], axis=0)
            arr_r_1 = np.concatenate([arr_r_1, ids[:, -2][:, np.newaxis], ids[:, -1][:, np.newaxis]], axis=1)
            arr_b_1 = np.concatenate([ids[0, :][np.newaxis, :], ids[1, :][np.newaxis, :], arr_b_1], axis=0)
            arr_l_1 = np.concatenate([ids[:, 0][:, np.newaxis], segment[:, 1][:, np.newaxis], arr_l_1], axis=1)

            ids[np.logical_and(ids == -1, arr_t != -1)] = arr_t[np.logical_and(ids == -1, arr_t != -1)]
            ids[np.logical_and(ids == -1, arr_r != -1)] = arr_r[np.logical_and(ids == -1, arr_r != -1)]
            ids[np.logical_and(ids == -1, arr_b != -1)] = arr_b[np.logical_and(ids == -1, arr_b != -1)]
            ids[np.logical_and(ids == -1, arr_l != -1)] = arr_l[np.logical_and(ids == -1, arr_l != -1)]

            ids[np.logical_and(ids == -1, arr_t_1 != -1)] = arr_t_1[np.logical_and(ids == -1, arr_t_1 != -1)]
            ids[np.logical_and(ids == -1, arr_r_1 != -1)] = arr_r_1[np.logical_and(ids == -1, arr_r_1 != -1)]
            ids[np.logical_and(ids == -1, arr_b_1 != -1)] = arr_b_1[np.logical_and(ids == -1, arr_b_1 != -1)]
            ids[np.logical_and(ids == -1, arr_l_1 != -1)] = arr_l_1[np.logical_and(ids == -1, arr_l_1 != -1)]

        segment[segment == -1] = 0
        ids[ids == -1] = 0

        if src_sens_path.stem in scene_specific_fixes_objectid:
            for ob_id in scene_specific_fixes_objectid[src_sens_path.stem]:
                segment[ids == ob_id] = scene_specific_fixes_objectid[src_sens_path.stem][ob_id]

        reduce_map, fold_map = get_reduce_and_fold_map()
        segment = fold_map[reduce_map[segment.flatten()]].reshape(segment.shape).astype(np.int8)
        semantic_bg = np.isin(segment, bg_classes)
        ids[semantic_bg] = 0

        alpha = 0.75

        ids = cv2.medianBlur(ids.astype(np.uint8), 5)
        segment = cv2.medianBlur(segment.astype(np.uint8), 5)

        boundaries_semantics = get_boundary_mask(segment)
        boundaries_instance = get_boundary_mask(ids)

        segment = (distinct_colors.get_color_fast_numpy(segment.reshape(-1)).reshape(list(segment.shape) + [3]) * 255).astype(np.uint8)
        ids = (distinct_colors.get_color_fast_numpy(ids.reshape(-1), override_color_0=True).reshape(list(ids.shape) + [3]) * 255).astype(np.uint8)
        color_image = np.array(Image.open(root / "rgb" / f"{item.stem}.jpg"))

        segment = segment * alpha + color_image * (1 - alpha)
        ids = ids * alpha + color_image * (1 - alpha)
        segment[boundaries_semantics > 0, :] = 0
        ids[boundaries_instance > 0, :] = 0

        Image.fromarray(ids.astype(np.uint8)).save(out_ins / item.name)
        Image.fromarray(segment.astype(np.uint8)).save(out_sem / item.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scannet preprocessing')
    parser.add_argument('--sens_root', required=False, default='/cluster/gimli/ysiddiqui/scannet_val_scans/scene0050_02', help='sens file path')
    parser.add_argument('--n', required=False, type=int, default=1, help='num proc')
    parser.add_argument('--p', required=False, type=int, default=0, help='current proc')
    args = parser.parse_args()

    sens_root = Path(args.sens_root)
    dest = Path("data/scannet/", sens_root.stem)
    dest.mkdir(exist_ok=True)
    print('#' * 80)
    print(f'extracting sens from {str(sens_root)} to {str(dest)}...')
    extract_scan(sens_root, dest)
    extract_labels(sens_root, dest)
    print('#' * 80)
    print('subsampling...')
    subsample_scannet_blur_window(dest, min_frames=900)
    visualize_labels(dest)
    print('#' * 80)
    print('mapping labels...')
    fold_scannet_classes(dest)
    visualize_mask_folder(dest / "rs_semantics")
    print('#' * 80)
    print('renumbering instances...')
    renumber_instances(dest)
    visualize_mask_folder(dest / "rs_instance")
    print('#' * 80)
    print('please run the following command for mask2former to generate machine generated panoptic segmentation')
    print(
        f'python demo.py --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml --input {str(dest.absolute())}/color --output {str(dest.absolute())}/panoptic --predictions {str(dest.absolute())}/panoptic --opts MODEL.WEIGHTS ../checkpoints/model_final_f07440.pkl')
    print('#' * 80)
    print('mapping coco labels...')
    map_panoptic_coco(dest)
    visualize_mask_folder(dest / "m2f_semantics")
    visualize_mask_folder(dest / "m2f_instance")
    visualize_mask_folder(dest / "m2f_segments")
    print('creating validation set')
    print('#' * 80)
    print('creating validation set')
    create_validation_set(dest, 0.20)
    print('#' * 80)
