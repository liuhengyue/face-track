import os, sys
import cv2
import math
import glob
import json
import numpy as np
import torch
from deepface import DeepFace
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from tqdm import tqdm


class FaceTrack:
    def __init__(self,
                 ref_image_path,
                 video_path,
                 tracker,
                 model_name="VGG-Face",
                 detector_backend="retinaface",
                 enforce_detection=False,
                 t_distance=0.5, # the threshold for verification
                 video_stride=5 # we roughly scan through the whole video to find a closest face
                 ):
        self.ref_image_path = ref_image_path
        self.video_path = video_path
        # a folder to store pre-splitted scene videos
        self.scene_paths = self.video_path.split(".")[0] + "_scenes/"
        # the output path to store final video clips
        self.output_clip_path = os.path.join(os.path.dirname(self.video_path), "clips")
        # create the paths
        os.makedirs(self.scene_paths, exist_ok=True)
        os.makedirs(self.output_clip_path, exist_ok=True)
        # define the output clip file name pattern
        self.output_clip_pattern = "clip-{:04d}"
        # the output path to store the final metadata
        self.output_metadata_path = os.path.join(os.path.dirname(self.video_path), "metadata.json")
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.t_distance = t_distance
        self.video_stride = video_stride

        self.ref_image = cv2.imread(ref_image_path) # bgr
        self.candidate_faces = []

        _ = DeepFace.build_model(self.model_name, task="facial_recognition")
        _ = DeepFace.build_model(self.detector_backend, task="face_detector")


        self.tracker = tracker

    def distance_to_ref_image(self, ref_face, face):
        if face['confidence'] < 0.1: # no face detected
            return 1.0
        return DeepFace.verify(ref_face,
                              face['face'],
                              model_name=self.model_name,
                              detector_backend=self.detector_backend,
                              enforce_detection=self.enforce_detection)['distance']

    def load_frame(self, cap, frame_id):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        return ret, frame

    def detect_from_frame(self, ref_face, cap, frame_id):
        ret, frame = self.load_frame(cap, frame_id)
        if not ret:
            return (1.0, None)
        faces = DeepFace.extract_faces(img_path=frame,
                                       detector_backend=self.detector_backend,
                                       enforce_detection=self.enforce_detection,
                                       align=True
                                       )
        face_distances = [self.distance_to_ref_image(ref_face, face) for face in faces]
        argmin_distance = np.argmin(face_distances)
        min_distance = face_distances[argmin_distance]
        candidate_face = faces[argmin_distance]
        return (min_distance, candidate_face)



    def find_face(self):
        """
        We find the closest face to the reference image in the whole video.
        Assumption:
         1. the target face exists in the video.
         2. the reference face is a "good" face.
        :return:
        """
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames: ", total_frames)
        min_distance = 1.1
        matched_face = None
        matched_frame_id = 0
        for frame_id in tqdm(range(0, total_frames, self.video_stride), desc="face detect in video"):
            distance, candidate_face = self.detect_from_frame(self.ref_image, cap, frame_id)
            if distance < min_distance:
                min_distance = distance
                matched_face = candidate_face
                matched_frame_id = frame_id
        self.candidate_faces.append(matched_face['face']) # it will be used for finding face for tracking


    def detect_scenes(self, split=True):
        # Open our video, create a scene manager, and add a detector.
        video = open_video(self.video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=27))
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()
        if split:
            split_video_ffmpeg(self.video_path, scene_list, output_dir=self.scene_paths, show_progress=True)
        return scene_list

    def get_all_videos(self):
        # List of possible video file extensions
        video_extensions = ["mp4", "avi", "mov", "mkv", "flv", "wmv"]
        video_files = []

        for ext in video_extensions:
            pattern = os.path.join(self.scene_paths, f"*.{ext}")
            video_files.extend(glob.glob(pattern))

        return video_files

    def process_scene(self):
        self.find_face()
        scene_list = self.detect_scenes(split=True)
        if len(scene_list):
            scene_video_paths = self.get_all_videos()
            scene_video_paths.sort()
            start_frame_ids = [s.frame_num for (s, e) in scene_list]
        else:
            scene_video_paths = [self.video_path]
            start_frame_ids = [0]
        # scene_video_paths = scene_video_paths[13:]
        ref_face = self.candidate_faces[0] if len(self.candidate_faces) else self.ref_image
        # store the output metadata (and the video)
        outputs = [] # file_name : annotation
        scene_id = 0
        # start_frame_id would be the global frame idx
        for scene_video_path, start_frame_id in zip(scene_video_paths, start_frame_ids):
            print("Finding target face in {}".format(scene_video_path))
            # load the video
            cap = cv2.VideoCapture(scene_video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            frame_id = 0
            found = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # detect the first matched face
                min_distance, face = self.detect_from_frame(ref_face, cap, frame_id)
                # print(min_distance)
                if min_distance < self.t_distance:
                    found = True
                    target_face_area = face['facial_area']
                    x1 = target_face_area['x']
                    y1 = target_face_area['y']
                    x2 = x1 + target_face_area['w']
                    y2 = y1 + target_face_area['h']
                    bbox = (x1, y1, x2, y2)
                    break
                frame_id += 1
            # if we found the face, start tracking
            if found:
                # output dict for this scene
                object_id = 0
                scene_path, ext = scene_video_path.split(".")

                frames = []
                face_coordinates = []
                clip_start_frame_id = start_frame_id + frame_id # global
                # actual time would be
                clip_start_timestamp = clip_start_frame_id / frame_rate
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    state = self.tracker.init_state(scene_video_path, offload_video_to_cpu=True)
                    _, _, masks = self.tracker.add_new_points_or_box(state, box=bbox, frame_idx=frame_id, obj_id=object_id)
                    # frame_idx would be local index for this scene
                    for frame_idx, object_ids, masks in self.tracker.propagate_in_video(state, start_frame_idx=frame_id):
                        mask_to_vis = {}
                        bbox_to_vis = {}
                        for obj_id, mask in zip(object_ids, masks):
                            mask = mask[0].cpu().numpy()
                            mask = mask > 0.0
                            non_zero_indices = np.argwhere(mask)
                            if len(non_zero_indices) == 0:
                                bbox = [0, 0, 0, 0]
                            else:
                                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                            bbox_to_vis[obj_id] = bbox
                            mask_to_vis[obj_id] = mask
                        # we should only have the object_id == 0?
                        bbox_tracked = bbox_to_vis[object_id]
                        is_tracked = sum(bbox_tracked) != 0
                        if is_tracked:
                            # we found a new clip
                            if clip_start_frame_id is None:
                                clip_start_frame_id = start_frame_id + frame_idx
                                clip_start_timestamp = clip_start_frame_id / frame_rate
                            face_coordinates.append(bbox_tracked)
                            _, loaded_frame = self.load_frame(cap, frame_idx)
                            frames.append(loaded_frame)
                        else:
                            # store current scene
                            if len(face_coordinates):
                                scene_output_path = "{}/{}.{}".format(self.output_clip_path, self.output_clip_pattern.format(scene_id), ext)  # to store only clips containing the target face
                                clip_end_frame_id = start_frame_id + frame_idx
                                clip_end_timestamp = clip_end_frame_id / frame_rate
                                assert clip_end_frame_id >= clip_start_frame_id
                                outputs.append({"file_name": scene_output_path,
                                                "start_timestamp": clip_start_timestamp,
                                                "end_timestamp": clip_end_timestamp,
                                                "start_frame_idx": clip_start_frame_id,
                                                "end_frame_idx": clip_end_frame_id,
                                                "face_coordinates": face_coordinates,
                                                })
                                # create the video clip
                                height, width = frames[0].shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                out = cv2.VideoWriter(scene_output_path, fourcc, frame_rate, (width, height))
                                for img in frames:
                                    out.write(img)
                                out.release()
                                # reset a scene
                                scene_id += 1
                                face_coordinates = []
                                frames = []
                                clip_start_frame_id = None
                    # if the whole video is tracked
                    if len(face_coordinates):
                        scene_output_path = "{}/{}.{}".format(self.output_clip_path, self.output_clip_pattern.format(scene_id), ext)
                        clip_end_frame_id = start_frame_id + frame_idx
                        clip_end_timestamp = clip_end_frame_id / frame_rate
                        assert clip_end_frame_id >= clip_start_frame_id
                        outputs.append({"file_name": scene_output_path,
                                        "start_timestamp": clip_start_timestamp,
                                        "end_timestamp": clip_end_timestamp,
                                        "start_frame_idx": clip_start_frame_id,
                                        "end_frame_idx": clip_end_frame_id,
                                        "face_coordinates": face_coordinates,
                                        })
                        # create the video clip
                        height, width = frames[0].shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(scene_output_path, fourcc, frame_rate, (width, height))
                        for img in frames:
                            out.write(img)
                        out.release()
                        scene_id += 1  # if the whole vid tracked, scene_id need to be increased
            # no found, then skip this scene

        # end of all scenes
        json_path = os.path.join(self.output_metadata_path)
        with open(json_path, 'w') as f:
            json.dump(outputs, f)
