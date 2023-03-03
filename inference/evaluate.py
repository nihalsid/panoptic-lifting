import argparse

from dataset.preprocessing.preprocess_scannet import calculate_iou_folders, calculate_panoptic_quality_folders
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='metrics')
    parser.add_argument('--root_path', required=False, default='data/scannet/scene0423_02')
    parser.add_argument('--exp_path', required=False, default='runs/scene0423_02_test_01170151_PanopLi_scannet042302_electrical-forest')
    args = parser.parse_args()

    print('calculating metrics for ours')
    image_dim = (512, 512)
    iou = calculate_iou_folders(Path(args.exp_path, "pred_semantics"), Path(args.root_path) / "rs_semantics", image_dim)
    pq, rq, sq = calculate_panoptic_quality_folders(Path(args.exp_path, "pred_semantics"), Path(args.exp_path, "pred_surrogateid"), Path(args.root_path) / "rs_semantics", Path(args.root_path) / "rs_instance", image_dim)
    print(f'[dataset] iou, pq, sq, rq: {iou:.3f}, {pq:.3f}, {sq:.3f}, {rq:.3f}')
