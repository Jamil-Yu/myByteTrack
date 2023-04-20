import cv2
import torch
cap=cv2.VideoCapture("/home/workspace/ByteTrack/videos/mot17.mp4")
ret_val, img = cap.read()
cv2.imwrite("test.jpg", img)
img = torch.from_numpy(img).unsqueeze(0)
from detectors.ultralytics import YOLO
detector = YOLO('/home/workspace/ByteTrack/detectors/yolov8m.pt')
img=img.view(1, 3, 1080, 1920)
dets = detector("test.jpg")
# print(dets[0].boxes.xyxy.shape)
# print(dets[0].boxes.conf.unsqueeze(-1).shape)
# print(torch.cat([dets[0].boxes.xyxy,dets[0].boxes.conf.unsqueeze(-1)],1))
print(dets[0].boxes)