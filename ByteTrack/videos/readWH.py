import cv2

 
video_path = "/home/workspace/ByteTrack/videos/MOT17.webm"

 
cap = cv2.VideoCapture(video_path)

frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

 
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
ret,frame=cap.read()
print(frame_height, frame_width)
print(ret)
cap.release()