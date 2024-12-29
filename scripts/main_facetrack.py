import argparse
import os.path as osp
import sys
# face detection and tracking
from src.face_track import FaceTrack
# tracking
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def build_tracker(model_path):
    model_cfg = determine_model_cfg(model_path)
    predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
    return predictor

def main(args):
    tracker = build_tracker(args.model_path)
    # run face detection before tracking
    face_detector = FaceTrack(args.ref_path, args.video_path, tracker)
    # face_detector.detect_scenes()
    face_detector.process_scene()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--ref_path", default=None, help="Path to the reference image (a face crop).")
    args = parser.parse_args()
    main(args)