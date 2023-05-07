from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import time
import threading
from PyQt5.QtWidgets import QApplication, QProgressDialog


from loguru import logger
import sys
import cv2
import torch
from my_predictor import Predictor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

import argparse
import os
import time


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    # args
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="webcam", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument(
        "--detector",default="yolov8", help="choose your detector, eg. yolox-s, yolox-m, yolox-l, yolox-x"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="/home/jamil/files/Git/myByteTrack/ByteTrack/videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="/home/jamil/files/Git/myByteTrack/ByteTrack/exps/my_exp.py",
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default="/home/jamil/files/Git/myByteTrack/ByteTrack/pretrained/bytetrack_x_mot20.tar", type=str, help="checkpoint for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.1, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


class Ui_MainWindow(object):
    # UI界面的代码
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1838, 1016)
        MainWindow.setStyleSheet("background-color: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(116, 193, 252, 255), stop:1 rgba(255, 255, 255, 255));\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.camera = QtWidgets.QLabel(self.centralwidget)
        self.camera.setGeometry(QtCore.QRect(20, 210, 891, 691))
        self.camera.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.camera.setObjectName("camera")
        self.detection = QtWidgets.QLabel(self.centralwidget)
        self.detection.setGeometry(QtCore.QRect(920, 210, 881, 691))
        self.detection.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.detection.setObjectName("detection")
        self.Begin = QtWidgets.QPushButton(self.centralwidget)
        self.Begin.setGeometry(QtCore.QRect(140, 130, 89, 25))
        self.Begin.setMinimumSize(QtCore.QSize(0, 25))
        self.Begin.setStyleSheet("background-color: rgba(153, 193, 241, 0);")
        self.Begin.setObjectName("Begin")
        self.Pause = QtWidgets.QPushButton(self.centralwidget)
        self.Pause.setGeometry(QtCore.QRect(260, 130, 89, 25))
        self.Pause.setStyleSheet("background-color: rgba(153, 193, 241, 0);")
        self.Pause.setObjectName("Pause")
        self.Intro = QtWidgets.QLabel(self.centralwidget)
        self.Intro.setGeometry(QtCore.QRect(1440, 170, 301, 16))
        self.Intro.setStyleSheet("")
        self.Intro.setObjectName("Intro")
        self.Caption = QtWidgets.QLabel(self.centralwidget)
        self.Caption.setGeometry(QtCore.QRect(530, 30, 741, 131))
        self.Caption.setStyleSheet("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 235, 235, 206), stop:0.35 rgba(188, 234, 255, 80), stop:0.4 rgba(137, 214, 255, 80), stop:0.425 rgba(132, 208, 255, 156), stop:0.44 rgba(128, 252, 247, 80), stop:1 rgba(255, 255, 255, 0));")
        self.Caption.setObjectName("Caption")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(20, 130, 86, 25))
        self.comboBox.setStyleSheet("background-color: rgba(153, 193, 241, 0);\n"
"selection-background-color: rgb(153, 193, 241);")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 96, 91, 21))
        self.label.setStyleSheet("background-color: rgba(255, 255, 255, 0%);")
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1838, 28))
        self.menubar.setObjectName("menubar")
        self.menuPRML_2 = QtWidgets.QMenu(self.menubar)
        self.menuPRML_2.setObjectName("menuPRML_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuPRML_2.addSeparator()
        self.menubar.addAction(self.menuPRML_2.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        # designer基础上，为实现UI界面的功能，添加的代码
        # 连接                                                
        self.Begin.clicked.connect(self.begin_clicked)       
        self.Pause.clicked.connect(self.pause_clicked)
        # 设置图片大小，自适应                                    
        self.camera.setScaledContents(True)                  
        self.detection.setScaledContents(True)               
        
    # UI界面的代码
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.camera.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">camera</p></body></html>"))
        self.detection.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">detection</p></body></html>"))
        self.Begin.setText(_translate("MainWindow", "Begin"))
        self.Pause.setText(_translate("MainWindow", "Pause"))
        self.Intro.setText(_translate("MainWindow", "Work of Jian Yu, Yifei Zhang, Shing-Ho Lin"))
        self.Caption.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; color:#241f31;\">PRML-课程展示-多目标跟踪系统</span></p></body></html>"))
        self.comboBox.setItemText(0, _translate("MainWindow", "yolov8"))
        self.comboBox.setItemText(1, _translate("MainWindow", "yolox"))
        self.label.setText(_translate("MainWindow", "检测模型选择"))
        self.menuPRML_2.setTitle(_translate("MainWindow", "PRML"))

    # designer基础上，为实现UI界面的功能，添加的代码
    def begin_clicked(self):
        selected_detector = self.comboBox.currentText()
        self.pause = False
        self.Camera_thread = CameraThread()  # 创建线程
        self.Detection_thread = DetectionThread(self.Camera_thread, selected_detector) # 创建线程，传入摄像头线程（因为不能同时打开摄像头）
        self.Camera_thread.changePixmap.connect(self.set_pixmap)
        self.Detection_thread.changeDetectionPixmap.connect(self.set_detection_pixmap)
        self.Camera_thread.start()
        self.Detection_thread.start()


    def set_pixmap(self, pixmap):
        self.camera.setPixmap(pixmap)
        self.camera.setScaledContents(True)


    def set_detection_pixmap(self, pixmap):
        self.detection.setPixmap(pixmap)
        self.detection.setScaledContents(True)

    def pause_clicked(self):
        self.pause = not self.pause

class CameraThread(QtCore.QThread):
    def __init__(self):
        super(CameraThread, self).__init__()
        self.ret = None
        self.frame = None

    changePixmap = QtCore.pyqtSignal(QtGui.QPixmap)


    def run(self):
        cap = cv2.VideoCapture(0)
        cap.open(0)
        while True:
            ret, frame = cap.read()
            if not ret:  # 如果没有读取到数据则跳过
                continue
            self.ret = ret
            self.frame = frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转成rgb形式
            h, w, ch = rgb_frame.shape  # 获取scale
            bytes_per_line = ch * w

            # 转成QImage格式，在界面上显示
            qimg = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.changePixmap.emit(pixmap)    

            if ui.pause:
                break

            # time.sleep(0.03)

        cap.release()

class DetectionThread(QtCore.QThread):
    def __init__(self, Camera_thread, selected_detector):
        super(DetectionThread, self).__init__()
        self.camera_thread = Camera_thread
        self.selected_detector = selected_detector


    changeDetectionPixmap = QtCore.pyqtSignal(QtGui.QPixmap)
    def run(self):
        dialog = QProgressDialog("正在加载中...", None, 0, 0)
        dialog.show()
        # 加载模型
        args = make_parser().parse_args()
        args.detector = self.selected_detector
        exp = get_exp(args.exp_file, args.name)
        predictor,current_time,args=main(exp, args)
        tracker = BYTETracker(args, frame_rate=30)
        timer = Timer()
        frame_id = 0
        results = []
        dialog.close()
        # 开始检测:
        while True:
            if frame_id % 1 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            ret = self.camera_thread.ret
            frame = self.camera_thread.frame

            if not ret:  # 如果没有读取到数据则跳过
                continue
            
            outputs, img_info = predictor.inference(frame, timer)
            tracker.isyolox=(predictor.exp._model == "yolox")
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                
                online_tlwhs = []
                online_ids = []
                online_scores = []
                
                for t in online_targets:
                    tlwh = t.tlwh#tlwh型（x_min,y_min,width,height）
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                
                results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
                timer.toc()
                
                online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                          fps=1. / timer.average_time)
            
            else:
                timer.toc()
                online_im = img_info['raw_img']

            rgb_frame = cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB)  # 转成rgb形式
            h, w, ch = rgb_frame.shape  # 获取scale
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

            # 把处理后的图像展示在 detection Label 上
            
            detection_pixmap = QtGui.QPixmap.fromImage(qimg)
            self.changeDetectionPixmap.emit(detection_pixmap)
            frame_id += 1
            if ui.pause:
                break

            # time.sleep(0.03)

def main(exp, args):
    torch.cuda.set_device('cuda:0')
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)


    if args.detector != "yolox":
        exp.get_model_from_args(args)
    
    
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        model.eval()

    if not args.trt and args.detector == "yolox":
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"], strict=False)
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)
    
    if args.fp16:
            model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    if args.detector == 'yolox':
        predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)#model is detector model
    else:
        predictor = Predictor(None, exp, trt_file, None, args.device, args.fp16)
    current_time = time.localtime()
    return predictor, current_time, args
    imageflow_demo(predictor, current_time, args)

if __name__ == "__main__":
    
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
