#! /usr/bin/python3
# coding=utf-8

import os
from typing import List, Optional


import cv2
import numpy as np
import torch
from .inference import auto_downsample_ratio, convert_video
from .inference_utils import ImageSequenceWriter
from .model import MattingNetwork
from .convert_video_onnx import convert_video_onnx

try:
	import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
	ort = None

_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "work", "checkpoint")
_ONNX_MODEL_PATH = os.path.join(_CHECKPOINT_DIR, "rvm_mobilenetv3.onnx")
_ONNX_MODEL_PATH_FP16 = os.path.join(_CHECKPOINT_DIR, "rvm_mobilenetv3_fp16.onnx")
_PTH_MODEL_PATH = os.path.join(_CHECKPOINT_DIR, "rvm_mobilenetv3.pth")


class RVMOnnxSession:
	"""Wrap ONNX Runtime inference for RVM."""

	def __init__(self, model_path: str, providers: Optional[List[str]] = None):
		if ort is None:
			raise RuntimeError("onnxruntime is required but not installed.")
		if providers is None:
			providers = ['CPUExecutionProvider']
		self.session = ort.InferenceSession(model_path, providers=providers)

		src_input = next((inp for inp in self.session.get_inputs() if inp.name == 'src'), None)
		if src_input is None:
			raise ValueError("ONNX model is missing 'src' input")
		self.src_dtype = np.float16 if 'float16' in src_input.type else np.float32

	def init_states(self):
		return [np.zeros((1, 1, 1, 1), dtype=self.src_dtype) for _ in range(4)]

	def infer(self, frame_rgb: np.ndarray, rec, downsample_ratio: float):
		if rec is None:
			rec = self.init_states()
		else:
			rec = [state if state is not None else np.zeros((1, 1, 1, 1), dtype=self.src_dtype) for state in rec]

		frame = frame_rgb.astype(self.src_dtype, copy=False)
		if frame.max() > 1.0:
			frame = frame / 255.0
		src = np.transpose(frame, (2, 0, 1))[np.newaxis, ...]  # [1, C, H, W]
		downsample = np.asarray([downsample_ratio], dtype=np.float32)

		inputs = {
			'src': src,
			'r1i': rec[0],
			'r2i': rec[1],
			'r3i': rec[2],
			'r4i': rec[3],
			'downsample_ratio': downsample
		}
		fgr, pha, r1o, r2o, r3o, r4o = self.session.run(None, inputs)
		return fgr, pha, [r1o, r2o, r3o, r4o]


# Module-level cache to avoid reloading heavy RVM checkpoint multiple times
_RVM_MODEL_CACHE = {}

def load_rvm_model(device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
	"""Load RVM model for silhouette extraction (singleton per device)."""
	onnx_path = None
	if ort is not None:
		if os.path.exists(_ONNX_MODEL_PATH_FP16):
			onnx_path = _ONNX_MODEL_PATH_FP16
		elif os.path.exists(_ONNX_MODEL_PATH):
			onnx_path = _ONNX_MODEL_PATH
	if onnx_path is not None:
		print(f"[RVM] Using ONNX Runtime (CPU)")
		device_key = f'onnx_cpu:{os.path.basename(onnx_path)}'
		cached_model = _RVM_MODEL_CACHE.get(device_key)
		if cached_model is not None:
			return cached_model
		model = RVMOnnxSession(onnx_path, providers=['CPUExecutionProvider'])
		_RVM_MODEL_CACHE[device_key] = model
		return model

	# Fallback to the legacy PyTorch model
	device_obj = torch.device(device)
	device_key = f"{device_obj.type}:{device_obj.index}" if device_obj.index is not None else device_obj.type

	cached_model = _RVM_MODEL_CACHE.get(device_key)
	if cached_model is not None:
		return cached_model

	model = MattingNetwork('mobilenetv3')
	state_dict = torch.load(_PTH_MODEL_PATH, map_location=device_obj)
	model.load_state_dict(state_dict)
	model = model.eval().to(device_obj)
	_RVM_MODEL_CACHE[device_key] = model
	return model


def calc_input_resize(video_path, frame_size_threshold=800):
	"""Calculate input resize for video."""
	cap = cv2.VideoCapture(video_path)
	frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	cap.release()
	if max(frame_width, frame_height) <= frame_size_threshold:
		return None
	if frame_width > frame_height:
		ratio = frame_width / frame_size_threshold
	else:
		ratio = frame_height / frame_size_threshold
	return (int(frame_width // ratio), int(frame_height // ratio))


def extract_silhouette_from_frame(model, frame_bgr, rec, input_resize=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
	"""Extract silhouette from a single frame using RVM model."""
	if input_resize is not None:
		frame_bgr = cv2.resize(frame_bgr, input_resize, interpolation=cv2.INTER_AREA)
	frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

	if isinstance(model, RVMOnnxSession):
		if rec is None:
			rec = model.init_states()
		h, w = frame_rgb.shape[:2]
		d_ratio = auto_downsample_ratio(h, w)
		_, pha, rec = model.infer(frame_rgb, rec, d_ratio)
		pha_np = (pha.squeeze((0, 1)) * 255.0).astype(np.uint8)
		return pha_np, rec

	# PyTorch fallback
	t = torch.from_numpy(frame_rgb).float() / 255.0  # H, W, C
	t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)  # 1,1,3,H,W

	with torch.no_grad():
		_, _, _, H, W = t.shape
		d_ratio = auto_downsample_ratio(H, W)
		fgr, pha, *rec = model(t, *rec, downsample_ratio=d_ratio)

	pha_np = (pha[0, 0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
	return pha_np, rec


def person_ext_rvm(vid, input_path, person_folder, frame_size_threshold=800):
	print(f"\t Start silhouette extraction.")

	silhouette_path = os.path.sep.join([person_folder, 'silhouette', vid])
	image_path = os.path.sep.join([person_folder, 'image', vid])
	os.makedirs(silhouette_path, exist_ok=True)
	os.makedirs(image_path, exist_ok=True)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = load_rvm_model(device)
	input_resize = calc_input_resize(input_path, frame_size_threshold)
	print(f"\t {input_resize=}")

	if isinstance(model, RVMOnnxSession):
		convert_video_onnx(
			model,
			input_source=input_path,
			input_resize=input_resize,
			output_composition=image_path,
			output_alpha=silhouette_path
		)
	else:
		convert_video(
			model,
			input_source=input_path,
			# num_workers=1,
			input_resize=input_resize,
			output_type='png_sequence',
			output_background='default',
			output_composition=image_path,
			output_alpha=silhouette_path,
			downsample_ratio=None,
			seq_chunk=4,
			progress=True
		)


# if __name__ == '__main__':
# 	input_path = "data\\upload\\14\\video\\b2092bb2.avi"
# 	person_folder = "data\\upload\\14"
# 	vid = "b2092bb2"
# 	person_ext_rvm(vid, input_path, person_folder)
