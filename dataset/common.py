import numpy as np
import decord
from PIL import Image
from torchvision import transforms
from decord import VideoReader, cpu


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, 
               return_msg=False, resolution=224, 
               hd_num=4, padding=False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape
        
    # return frames
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames

def load_image(path_like):
    """
    Support for multiple image path input
    """
    if isinstance(path_like, str):
        try:
            image = Image.open(path_like).convert('RGB')
        except:
            image = None
    return image
