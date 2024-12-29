# FaceTrack

This repository is an implementation to detect and track the reference face from a video.

[Hengyue Liu](https://hengyueliu.com)

## Getting Started

#### Install Basics (works for cuda 11.8)
```
conda create --name=facetrack python=3.10
conda activate facetrack
pip install tensorflow[and-cuda]
pip install tf-keras
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install deepface
pip install scenedetect[opencv]
pip install loguru scipy
```

#### Install SAMURAI for tracking 

SAM2-based VOS method SAMURAI is used. SAM2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://github.com/facebookresearch/sam2?tab=readme-ov-file) to install both PyTorch and TorchVision dependencies. You can install **the SAMURAI version** of SAM 2 on a GPU machine using:
```
cd sam2
pip install -e ".[notebooks]"
```

Please see [INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) from the original SAM 2 repository for FAQs on potential issues and solutions.

#### Install a fix
Since the project uses both Tensorflow (cudnn 9.3) and PyTorch (9.1), need a compatible cudnn version after installing both:
```
pip uninstall nvidia-cudnn-cu11
pip install nvidia-cudnn-cu12==9.3.0.75
```

#### SAM 2.1 Checkpoint Download

```
cd checkpoints && \
./download_ckpts.sh && \
cd .. && \
cd ..
```

## Demo

#### Data Preparation

Please prepare the data in the following format:
```
data/
├── demo1/
│   ├── video
|   |── reference face image 
```

#### Main Inference
```
python scripts/main_facetrack.py --video_path data/demo1/top_gun.mp4 --ref_path data/demo1/tom_cruise.jpg
```
Note: Need at least 8GB memory to run successfully. We can relax this requirement by using smaller SAM2 or deepface model.

Current default setting uses SAM2 model`sam2.1_hiera_base_plus.pt`, deepface models `VGG-Face` for verification, and `retinaface` for face detection.

#### Data Preparation

After the run, you will have a folder `clips/` containing all crops containing the target face, with a json file `metadata.json` containing annotations of face coordinates and timestamps:
```
data/
├── demo1/
│   ├── video_file_name.mp4
|   |── reference_image.png 

### generated files ###
|   |── [video_file_name]_scenes/ # contains splitted scenes 
|   |── clips/ 
|   |   |── clip-0000.mp4 
|   |   |── clip-0001.mp4
|   |   |── ...
|   |── metadata.json
```

#### Data Preparation

Check the visualization of face bboxes on original video in file [visualization.ipynb](scripts/visualization.ipynb).

The corresponding annotated videos are stored as `demo.mp4`. One example shown here:


https://github.com/user-attachments/assets/794cc102-f5c0-4bd1-8069-5c69e8ad3edb



## Acknowledgment

[SAM 2](https://github.com/facebookresearch/sam2?tab=readme-ov-file)

[SAMURAI](https://github.com/yangchris11/samurai/tree/master)

[PySceneDetect](https://github.com/Breakthrough/PySceneDetect/tree/main)

[DeepFace](https://github.com/serengil/deepface/tree/master)


