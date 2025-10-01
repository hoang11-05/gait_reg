# import cv2 
# import numpy as np
# import time 

# np.random.seed(42)
# class Detector:
#     def __init__(self, videoPath, configPath, modelPath, classesPath):
#         self.video_path= videoPath
#         self.configPath= configPath
#         self.modelPath= modelPath        
#         self.classesPath= classesPath

#         self.net = cv2.dnn.DetectionModel(self.modelPath, self.configPath)
#         self.net.setInputSize(320, 320)
#         self.net.setInputScale(1.0 / 127.5)
#         self.net.setInputMean((127.5, 127.5, 127.5))
#         self.net.setInputSwapRB(True)
#         self.readClasses()
#     def readClasses(self):
#         with open(self.classesPath, 'r') as f:
#             self.classesList = f.read().splitlines()
#         self.classesList.insert(0, '__Background__')
#         print(self.classesList)
#         self.colors = np.random.uniform(0, 255, size=(len(self.classesList), 3))
    
#     def onVideo(self):
#         cap=cv2.VideoCapture(self.video_path)
#         if not cap.isOpened():
#             return
#         (success, img) = cap.read()

#         while success:
#             classLabelIDs, confidences, bboxs= self.net.detect(img, confThreshold=0.5)
#             bboxs= list(bboxs)
#             confidences= list(np.array(confidences).reshape(1, -1)[0])
#             confidences= list(map(float, confidences))
#             bboxIdx= cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)
#             if len(bboxIdx) != 0:
#                for i in range(len(bboxIdx)):
#                 bbox= bboxs[np.squeeze(bboxIdx[i])]
#                 classConfidence= confidences[np.squeeze(bboxIdx[i])]
#                 classLabelID= np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
#                 classLabel= self.classesList[classLabelID]
#                 classColor= self.colors[classLabelID]
#                 displayText= "{}:{:.2f}".format(classLabel, classConfidence)
               
#                 x,y,w,h= bbox
#                 cv2.rectangle(img, (x,y), (x+w, y+h), color=self.colors[classLabelID], thickness=2)
#                 cv2.putText(img, displayText, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, classColor, 2)


#                 lineWidth= 30
#                 cv2.line(img, (x, y), (x+lineWidth, y), classColor, thickness=5)
#                 cv2.line(img, (x, y), (x, y+lineWidth), classColor, thickness=5)
#                 cv2.line(img, (x+w, y), (x+w-lineWidth, y), classColor, thickness=5)
#                 cv2.line(img, (x+w, y), (x+w, y+lineWidth), classColor, thickness=5)
#                 cv2.line(img, (x, y+h), (x+lineWidth, y+h), classColor, thickness=5)
#                 cv2.line(img, (x, y+h), (x, y+h-lineWidth), classColor, thickness=5)
#                 cv2.line(img, (x+w, y+h), (x+w-lineWidth, y+h), classColor, thickness=5)
#                 cv2.line(img, (x+w, y+h), (x+w, y+h-lineWidth), classColor, thickness=5)
#             cv2.imshow("Result", img)

#             key= cv2.waitKey(1) & 0xFF
#             if key== ord('q'):
#                 break
#             (success, img) = cap.read()
#         cv2.destroyAllWindows()    

import cv2
import numpy as np

np.random.seed(42)

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.video_path = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Load SSD model
        self.net = cv2.dnn.DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Load classes
        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        self.classesList.insert(0, '__Background__')
        self.colors = np.random.uniform(0, 255, size=(len(self.classesList), 3))

    def detect_person(self, img, label=""):
        """
        Detect person trong frame, vẽ box và label (nếu chỉ có 1 người).
        """
        classIDs, confidences, boxes = self.net.detect(img, confThreshold=0.6, nmsThreshold=0.0)
        # nmsThreshold=0.0 để tự lọc sau

        if classIDs is None or len(classIDs) == 0:
            return img

        # Lấy ra bbox cho person
        person_boxes = []
        person_confs = []
        for class_id, confidence, box in zip(classIDs.flatten(), confidences.flatten(), boxes):
            if self.classesList[class_id].lower() == "person":
                person_boxes.append(box)
                person_confs.append(float(confidence))

        if len(person_boxes) == 0:
            return img

        # Áp dụng NMS để loại trùng box
        indices = cv2.dnn.NMSBoxes(person_boxes, person_confs, score_threshold=0.5, nms_threshold=0.1)

        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                x, y, w, h = person_boxes[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if label and len(indices) == 1:  # chỉ annotate khi có đúng 1 người
                    cv2.putText(img, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img

    def onVideo(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        success, img = cap.read()
        while success:
            img = self.detect_person(img)  
            cv2.imshow("Result", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            success, img = cap.read()

        cap.release()
        cv2.destroyAllWindows()
