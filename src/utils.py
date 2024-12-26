import numpy as np
import torch
from PIL import Image, ImageOps

def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    # img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    # img_pil = ImageOps.pad(img_pil, (image_size, image_size), color='black', centering=(0, 0)).convert("RGB")
    new_image = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    new_image.paste(img_pil.convert("RGB"), (0, 0))
    img_np = np.array(new_image)
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    original_width, original_height = img_pil.size  # the original video size
    return img, original_height, original_width

# modified from load_video_frames_from_jpg_images
def load_image(img_path,
               image_size,
               offload_video_to_cpu,
               img_mean=(0.485, 0.456, 0.406),
               img_std=(0.229, 0.224, 0.225),
               async_loading_frames=False,
               compute_device=torch.device("cuda"),
               ):
    img, original_height, original_width = _load_img_as_tensor(
        img_path, image_size
    )
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    if not offload_video_to_cpu:
        img = img.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    img -= img_mean
    img /= img_std
    return img, original_height, original_width