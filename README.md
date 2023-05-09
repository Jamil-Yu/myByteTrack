# myByteTrack
## 配置环境
和ByteTrack一样

```shell
git clone https://github.com/Jamil-Yu/myByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
```

同时为了支持ui界面，需要安装pyqt5

```shell
pip install pqyt5
```



## 运行
视频到视频

```shell
python3 tools/my_track.py video -f exps/my_exp.py -c pretrained/bytetrack_x_mot20.tar --fp16 --fuse --save_result --path /home/workspace/ByteTrack/videos/mot17.mp4 --detector yolov8
```

UI界面

```shell
python3 tools/Main_Window.py
```



## 解释
my_track.py为main文件

my_exp.py为配置文件

bytetrack_x_mot20.tar为训练好的模型

/home/workspace/ByteTrack/videos/mot17.mp4 为视频路径

yolov8 为detector,目前支持yolox,yolov8
