import os

conf = {
    "WORK_PATH": os.path.dirname(__file__),
    "CUDA_VISIBLE_DEVICES": "cpu",  # "0,1" "cpu"
    "ALLOWED_EXTENSIONS": {'mp4', 'avi', 'wmv', 'flv', 'mov'},
    "UPLOAD_FOLDER": os.path.sep.join([os.path.dirname(__file__), 'data', 'upload']),
    "TMP_FOLDER": os.path.sep.join([os.path.dirname(__file__), 'data', 'upload', 'tmp']),
    "DATASETS_FOLDER": os.path.sep.join([os.path.dirname(__file__), 'model', 'gait', 'datasets', 'real-pkl-64']),
    "PRE_METHOD": "rvm",  
    "ext_frame_size_threshold": 480,
    "cut_img_pixel_threshold": 120,
    # Fast RAM mode config for K-frame search
    "FAST_RAM": {
        "K_search_interval": 6,  # every K frames perform a search
        "window_frames": 64,     # size of sliding window (clip length)
        "min_clip": 6         # minimum frames before first search
    },
    # Recognition source defaults (editable)
    "RECOGNITION_SOURCE_DEFAULT": {
        "mode": "rtsp",       # options: "upload", "webcam", "rtsp"
        "webcam_index": 0,       # used when mode == "webcam"
        "rtsp_url": "rtsp://swadmin:234567cn@117.2.126.196:554/stream1"          # used when mode == "rtsp"
    },
    # Recognition source runtime state (managed in-app, do not edit manually)
    "RECOGNITION_SOURCE": None,
    # model gait recognition config
    "MODEL_CONFIG": r"D:\Gait_RECOGNITION\model\gait\configs\gaibase\casia-b\gaitbase_casia-b.yaml",
    "MODEL_CHECKPOINT": r"D:\Gait_RECOGNITION\model\gait\output\Casia-b\gaitbase\casia-b\checkpoints\GaitBase_DA-60000.pt",
}

if conf.get("RECOGNITION_SOURCE") is None:
    conf["RECOGNITION_SOURCE"] = dict(conf.get("RECOGNITION_SOURCE_DEFAULT", {}))