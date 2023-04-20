# myByteTrack
# 配置环境和ByteTrack一样
# 运行
python3 tools/my_track.py video -f exps/my_exp.py -c pretrained/bytetrack_x_mot20.tar --fp16 --fuse --save_result --path /home/workspace/ByteTrack/videos/mot17.mp4 --detector yolov8

# my_track.py为main文件
# my_exp.py为配置文件
# bytetrack_x_mot20.tar为训练好的模型
# /home/workspace/ByteTrack/videos/mot17.mp4 为视频路径
# yolov8 为detector,目前支持yolox,yolov8
