import cv2
import os

def crop_person(frame, bbox):
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]

def save_result(person_id, result, output_dir='output/person_tracking'):
    os.makedirs(output_dir, exist_ok=True)
    # Example: save silhouette or features
    # cv2.imwrite(f'{output_dir}/person_{person_id}.png', result)
    pass
