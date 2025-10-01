import cv2
import time
import collections

from model.person_det.SSD.model_data.detect_person import ssd_detect_person
from model.person_det.yolov5.detect_person import yolov5_detect_person
from model.person_det.yolov8.detect_person import yolov8_detect_person

model_detect_func = yolov5_detect_person  

def test_model_on_video(video_path, smooth_n=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    frame_count = 0
    frame_dt_deque = collections.deque(maxlen=smooth_n)   # pipeline time
    inf_dt_deque = collections.deque(maxlen=smooth_n)     # inference time
    prev_time = time.perf_counter()
    start_all = prev_time
    total_inf_time = 0.0
    valid_inf_frames = 0   # số frame inference hợp lệ

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # --- đo pipeline time ---
        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            frame_dt_deque.append(dt)

        # --- đo inference time ---
        inf_start = time.perf_counter()
        try:
            result_frame = model_detect_func(frame, label='')
        except Exception as exc:
            print(f"[Warning] Detect failed on frame {frame_count}: {exc}")
            result_frame = frame.copy()
        inf_end = time.perf_counter()

        inf_dur = inf_end - inf_start
        # chỉ tính nếu inference > 1ms (0.001s) để tránh số ảo
        if inf_dur > 0.001:
            inf_dt_deque.append(inf_dur)
            total_inf_time += inf_dur
            valid_inf_frames += 1

        # resize cho hiển thị
        result_frame = cv2.resize(result_frame, (640, 360))

        # --- tính FPS smoothed ---
        pipeline_fps = (len(frame_dt_deque) / sum(frame_dt_deque)) if frame_dt_deque else 0.0
        inf_fps = (len(inf_dt_deque) / sum(inf_dt_deque)) if inf_dt_deque else 0.0

        # --- tính FPS trung bình ---
        total_elapsed = time.perf_counter() - start_all
        avg_fps = (frame_count / total_elapsed) if total_elapsed > 0 else 0.0
        avg_inf_fps = (valid_inf_frames / total_inf_time) if total_inf_time > 0 else 0.0

        # vẽ thông tin lên khung
        cv2.putText(result_frame, f"Pipeline FPS: {pipeline_fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if inf_fps > 0:
            cv2.putText(result_frame, f"Inference FPS: {inf_fps:.2f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Avg FPS: {avg_fps:.2f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Result', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count} | pipeline_fps={pipeline_fps:.2f} "
                  f"inf_fps={inf_fps:.2f} avg_fps={avg_fps:.2f}")

    # --- in kết quả tổng ---
    end_all = time.perf_counter()
    total_fps = frame_count / (end_all - start_all) if (end_all - start_all) > 0 else 0.0
    avg_inf_fps_total = (valid_inf_frames / total_inf_time) if total_inf_time > 0 else 0.0
    print(f"Total frames: {frame_count}, "
          f"Average pipeline FPS: {total_fps:.2f}, "
          f"Average inference FPS: {avg_inf_fps_total:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"D:\gait_reg\vlc-record-2025-09-26-10h45m32s-rtsp___117.2.126.196_554_stream1-.avi"
    test_model_on_video(video_path)
