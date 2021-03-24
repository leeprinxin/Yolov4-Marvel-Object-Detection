import cv2 as cv, os, time
import numpy as np
import pafy

#載入yolov4
net = cv.dnn.readNet('yolov4-marvel.cfg', 'yolov4-marvel_4000.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

#載入yolov4-labels
with open('marvel.names', 'rt') as f:
	names = f.read().rstrip('\n').split('\n')

# 取得youtube影片
url = "https://www.youtube.com/watch?v=osSJhXruEzU"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

capture = cv.VideoCapture(best.url)
FPS = "Initialing"
frame_count = 0

color_map ={'Thanos':(51, 102, 153),
 'CaptainAmerica':(0, 51, 102),
 'IronMan': (255, 0, 0),
 'SpiderMan':(0, 0, 0)}


while capture.isOpened():
    ret, frame = capture.read()


    classes, confidences, boxes = model.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    if not classes == ():#有辨識才處理
      #对识别目标进行标注
      for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        if confidence>0.65:
            left, top, w, h = box
            cv.rectangle(frame, (left, top), (left + w, top + h), color_map[names[classId]], 2)
            cv.rectangle(frame, (left, top), (left + 200, top+20), color_map[names[classId]], -1)
            cv.putText(frame, '%.2f '%(confidence*100)+names[classId], (left, top+16), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2)

    # ----FPS count
    if frame_count == 0:
        t_start = time.time()
    frame_count += 1
    if frame_count >= 10:
        FPS = "FPS=%d" % (int(frame_count / (time.time() - t_start)))
        frame_count = 0

    cv.putText(frame, FPS, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv.putText(frame, 'GTX 1660s', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv.imshow('test',frame)
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
