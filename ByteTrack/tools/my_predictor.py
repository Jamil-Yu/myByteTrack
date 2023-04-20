import cv2
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yolox.data.data_augment import preproc
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
class Predictor(object):
    def __init__(
        self,
        model=None,
        exp=None,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False
    ):
        if model is not None:
            self.model = model
        else:
            self.model=exp.optional_detector

        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.exp=exp
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.img_path=exp.img_path

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        if self.exp._model == "yolov8":
            cv2.imwrite("store.jpg",img)
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        if self.exp._model == "yolox":
            img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
            img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16
        
        with torch.no_grad():
            timer.tic()
            if self.exp._model == "yolox":
                outputs = self.model(img)##这里是detector模型的输出
                if self.decoder is not None:
                    outputs = self.decoder(outputs, dtype=outputs.type())
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
            elif self.exp._model == "yolov8":
                
                # outputs_wicls = self.model("store.jpg")[0].boxes.xyxy
                outputs_wicls=torch.cat([self.model("store.jpg")[0].boxes.xyxy,self.model("store.jpg")[0].boxes.conf.unsqueeze(-1)],1)
                # outputs_clone=outputs_wicls.cpu().numpy()
                outputs=[outputs_wicls.cpu().numpy()]
                
            else:
                print(self.exp._model)
                raise NotImplementedError
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        #outputs,[left,top,right,bottom,obj_conf,cls_conf,cls_id]
        return outputs, img_info#返回的是detector模型的输出和图像信息