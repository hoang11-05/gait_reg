import cv2
import time
import collections


from model.person_det.SSD.model_data.detect_person import ssd_detect_person
from model.person_det.yolov5.detect_person import yolov5_detect_person
from model.person_det.yolov8.detect_person import yolov8_detect_person
from model.person_ext.rvm.person_ext import load_rvm_model, calc_input_resize, extract_silhouette_from_frame

model_detect_func = yolov5_detect_person 

# === YOLOv8 Segmentation (for silhouette extraction) ===
try:
    from ultralytics import YOLO as _YOLO
    _yolov8_seg_model = _YOLO("yolov8n-seg.pt")  # auto-download if missing
except Exception:
    _YOLO = None
    _yolov8_seg_model = None


def _extract_silhouette_yolov8seg(frame):
    """Return grayscale silhouette (0..255) extracted by YOLOv8 segmentation for class person.
    Fallback: return black image if model unavailable or no person mask.
    """
    if _yolov8_seg_model is None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * 0
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _yolov8_seg_model.predict(img_rgb, classes=[0], verbose=False)
    if not results:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * 0
    res = results[0]
    # Compose instance masks for person class if available
    mask = None
    try:
        if getattr(res, "masks", None) is not None and res.masks is not None:
            # res.masks.data: [N, H, W] boolean/float masks (tensor)
            masks_tensor = res.masks.data
            if masks_tensor is not None and masks_tensor.shape[0] > 0:
                # Combine all person masks
                import numpy as np
                combined = masks_tensor.any(dim=0).cpu().numpy().astype("uint8") * 255
                mask = combined
    except Exception:
        mask = None
    if mask is None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * 0
    # Ensure mask size equals frame size
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def test_model_on_video_with_ext(video_path, smooth_n=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    frame_count = 0
    frame_dt_deque = collections.deque(maxlen=smooth_n)
    inf_dt_deque = collections.deque(maxlen=smooth_n)
    ext_dt_deque = collections.deque(maxlen=smooth_n)
    ext_seg_dt_deque = collections.deque(maxlen=smooth_n)
    prev_time = time.perf_counter()
    start_all = prev_time
    total_inf_time = 0.0
    total_ext_time = 0.0
    total_ext_seg_time = 0.0
    valid_inf_frames = 0
    valid_ext_frames = 0
    valid_ext_seg_frames = 0

    # --- Khởi tạo RVM model và các tham số ---
    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    rvm_model = load_rvm_model(device)
    input_resize = calc_input_resize(video_path, frame_size_threshold=800)
    rec = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            frame_dt_deque.append(dt)

        # --- detect ---
        inf_start = time.perf_counter()
        try:
            result_frame = model_detect_func(frame, label='')
        except Exception as exc:
            print(f"[Warning] Detect failed on frame {frame_count}: {exc}")
            result_frame = frame.copy()
        inf_end = time.perf_counter()
        inf_dur = inf_end - inf_start
        if inf_dur > 0.001:
            inf_dt_deque.append(inf_dur)
            total_inf_time += inf_dur
            valid_inf_frames += 1

        # --- person_ext với RVM ---
        ext_start = time.perf_counter()
        try:
            pha_np, rec = extract_silhouette_from_frame(rvm_model, frame, rec, input_resize, device)
            ext_frame = cv2.cvtColor(pha_np, cv2.COLOR_GRAY2BGR)
        except Exception as exc:
            print(f"[Warning] Ext (RVM) failed on frame {frame_count}: {exc}")
            ext_frame = frame.copy()
        ext_end = time.perf_counter()
        ext_dur = ext_end - ext_start
        if ext_dur > 0.001:
            ext_dt_deque.append(ext_dur)
            total_ext_time += ext_dur
            valid_ext_frames += 1

        # --- person_ext với YOLOv8 Seg ---
        seg_start = time.perf_counter()
        try:
            seg_mask = _extract_silhouette_yolov8seg(frame)
            seg_frame = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
        except Exception as exc:
            print(f"[Warning] Ext (YOLOv8-Seg) failed on frame {frame_count}: {exc}")
            seg_frame = frame.copy()
        seg_end = time.perf_counter()
        seg_dur = seg_end - seg_start
        if seg_dur > 0.001:
            ext_seg_dt_deque.append(seg_dur)
            total_ext_seg_time += seg_dur
            valid_ext_seg_frames += 1

        # resize cho hiển thị
        result_frame = cv2.resize(result_frame, (640, 360))
        ext_frame = cv2.resize(ext_frame, (640, 360))
        seg_frame = cv2.resize(seg_frame, (640, 360))

        # --- tính FPS ---
        pipeline_fps = (len(frame_dt_deque) / sum(frame_dt_deque)) if frame_dt_deque else 0.0
        inf_fps = (len(inf_dt_deque) / sum(inf_dt_deque)) if inf_dt_deque else 0.0
        ext_fps = (len(ext_dt_deque) / sum(ext_dt_deque)) if ext_dt_deque else 0.0
        ext_seg_fps = (len(ext_seg_dt_deque) / sum(ext_seg_dt_deque)) if ext_seg_dt_deque else 0.0
        total_elapsed = time.perf_counter() - start_all
        avg_fps = (frame_count / total_elapsed) if total_elapsed > 0 else 0.0
        avg_inf_fps = (valid_inf_frames / total_inf_time) if total_inf_time > 0 else 0.0
        avg_ext_fps = (valid_ext_frames / total_ext_time) if total_ext_time > 0 else 0.0
        avg_ext_seg_fps = (valid_ext_seg_frames / total_ext_seg_time) if total_ext_seg_time > 0 else 0.0

        # vẽ thông tin lên khung
        cv2.putText(result_frame, f"Detect FPS: {inf_fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(ext_frame, f"RVM Ext FPS: {ext_fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(seg_frame, f"YOLOv8-Seg FPS: {ext_seg_fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        cv2.putText(result_frame, f"Avg FPS: {avg_fps:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(ext_frame, f"Avg RVM Ext FPS: {avg_ext_fps:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(seg_frame, f"Avg YOLOv8-Seg FPS: {avg_ext_seg_fps:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

        # hiển thị song song
        cv2.imshow('Detect Result', result_frame)
        cv2.imshow('Person Ext Result (RVM)', ext_frame)
        cv2.imshow('Person Ext Result (YOLOv8-Seg)', seg_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count} | detect_fps={inf_fps:.2f} rvm_ext_fps={ext_fps:.2f} y8seg_fps={ext_seg_fps:.2f} avg_fps={avg_fps:.2f}")

    # --- in kết quả tổng ---
    end_all = time.perf_counter()
    total_fps = frame_count / (end_all - start_all) if (end_all - start_all) > 0 else 0.0
    avg_inf_fps_total = (valid_inf_frames / total_inf_time) if total_inf_time > 0 else 0.0
    avg_ext_fps_total = (valid_ext_frames / total_ext_time) if total_ext_time > 0 else 0.0
    avg_ext_seg_fps_total = (valid_ext_seg_frames / total_ext_seg_time) if total_ext_seg_time > 0 else 0.0
    print(f"Total frames: {frame_count}, "
          f"Average detect FPS: {avg_inf_fps_total:.2f}, "
          f"Average RVM ext FPS: {avg_ext_fps_total:.2f}, "
          f"Average YOLOv8-Seg FPS: {avg_ext_seg_fps_total:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"D:\gait_reg\test_hoang.mp4"
    test_model_on_video_with_ext(video_path)
