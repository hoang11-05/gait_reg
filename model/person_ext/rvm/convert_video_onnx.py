import os
import cv2
import numpy as np
from tqdm import tqdm

def convert_video_onnx(model, input_source, input_resize=None, output_composition=None, output_alpha=None, progress=True):
    """
    ONNX version of convert_video for RVM. Only supports png_sequence output for now.
    Args:
        model: RVMOnnxSession instance
        input_source: video file path
        input_resize: (w, h) tuple or None
        output_composition: directory to save RGB frames
        output_alpha: directory to save alpha (silhouette) frames
        seq_chunk: number of frames to process at once (not used, for compatibility)
        progress: show progress bar
    """
    assert output_composition or output_alpha, 'Must provide at least one output.'
    assert os.path.isfile(input_source), 'Only video file input supported.'

    cap = cv2.VideoCapture(input_source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_composition, exist_ok=True) if output_composition else None
    os.makedirs(output_alpha, exist_ok=True) if output_alpha else None

    rec = None
    idx = 0
    pbar = tqdm(total=total_frames, disable=not progress, dynamic_ncols=True)
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if input_resize is not None:
            frame_bgr = cv2.resize(frame_bgr, input_resize, interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        d_ratio = min(512 / max(h, w), 1)
        _, pha, rec = model.infer(frame_rgb, rec, d_ratio)
        pha_np = (pha.squeeze((0, 1)) * 255.0).astype(np.uint8)
        if output_alpha:
            cv2.imwrite(os.path.join(output_alpha, f"{idx:04d}.png"), pha_np)
        if output_composition:
            cv2.imwrite(os.path.join(output_composition, f"{idx:04d}.png"), frame_bgr)
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
