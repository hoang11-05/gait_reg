import cv2
import time
import collections


from model.person_det.SSD.model_data.detect_person import ssd_detect_person
from model.person_det.yolov5.detect_person import yolov5_detect_person
from model.person_det.yolov8.detect_person import yolov8_detect_person
from model.person_ext.rvm.person_ext import load_rvm_model, calc_input_resize, extract_silhouette_from_frame

model_detect_func = yolov5_detect_person 

def test_model_on_video_with_ext(video_path, smooth_n=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    frame_count = 0
    frame_dt_deque = collections.deque(maxlen=smooth_n)
    inf_dt_deque = collections.deque(maxlen=smooth_n)
    ext_dt_deque = collections.deque(maxlen=smooth_n)
    prev_time = time.perf_counter()
    start_all = prev_time
    total_inf_time = 0.0
    total_ext_time = 0.0
    valid_inf_frames = 0
    valid_ext_frames = 0

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

        # --- person_ext đúng chuẩn ---
        ext_start = time.perf_counter()
        try:
            pha_np, rec = extract_silhouette_from_frame(rvm_model, frame, rec, input_resize, device)
            ext_frame = cv2.cvtColor(pha_np, cv2.COLOR_GRAY2BGR)
        except Exception as exc:
            print(f"[Warning] Ext failed on frame {frame_count}: {exc}")
            ext_frame = frame.copy()
        ext_end = time.perf_counter()
        ext_dur = ext_end - ext_start
        if ext_dur > 0.001:
            ext_dt_deque.append(ext_dur)
            total_ext_time += ext_dur
            valid_ext_frames += 1

        # resize cho hiển thị
        result_frame = cv2.resize(result_frame, (640, 360))
        ext_frame = cv2.resize(ext_frame, (640, 360))

        # --- tính FPS ---
        pipeline_fps = (len(frame_dt_deque) / sum(frame_dt_deque)) if frame_dt_deque else 0.0
        inf_fps = (len(inf_dt_deque) / sum(inf_dt_deque)) if inf_dt_deque else 0.0
        ext_fps = (len(ext_dt_deque) / sum(ext_dt_deque)) if ext_dt_deque else 0.0
        total_elapsed = time.perf_counter() - start_all
        avg_fps = (frame_count / total_elapsed) if total_elapsed > 0 else 0.0
        avg_inf_fps = (valid_inf_frames / total_inf_time) if total_inf_time > 0 else 0.0
        avg_ext_fps = (valid_ext_frames / total_ext_time) if total_ext_time > 0 else 0.0

        # vẽ thông tin lên khung
        cv2.putText(result_frame, f"Detect FPS: {inf_fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(ext_frame, f"Ext FPS: {ext_fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(result_frame, f"Avg FPS: {avg_fps:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(ext_frame, f"Avg Ext FPS: {avg_ext_fps:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # hiển thị song song
        cv2.imshow('Detect Result', result_frame)
        cv2.imshow('Person Ext Result', ext_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count} | detect_fps={inf_fps:.2f} ext_fps={ext_fps:.2f} avg_fps={avg_fps:.2f}")

    # --- in kết quả tổng ---
    end_all = time.perf_counter()
    total_fps = frame_count / (end_all - start_all) if (end_all - start_all) > 0 else 0.0
    avg_inf_fps_total = (valid_inf_frames / total_inf_time) if total_inf_time > 0 else 0.0
    avg_ext_fps_total = (valid_ext_frames / total_ext_time) if total_ext_time > 0 else 0.0
    print(f"Total frames: {frame_count}, "
          f"Average detect FPS: {avg_inf_fps_total:.2f}, "
          f"Average ext FPS: {avg_ext_fps_total:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"D:\gait_reg\vlc-record-2025-09-26-10h45m32s-rtsp___117.2.126.196_554_stream1-.avi"
    test_model_on_video_with_ext(video_path)
