# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp

from detectors.ultralytics import YOLO

model_sets=[
    'yolov8',
    'yolox'
]#detector models to choose from

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.warmup_epochs = 1
        self.test_size=(896, 1600)
        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.img_path=None
        self.optional_detector=None
        self._model="yolox"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
    
    def get_model_from_args(self,args):
        if args.detector in model_sets:
            self.img_path=args.path
            self._model=args.detector
            if args.detector=='yolov8':
                self.optional_detector=YOLO('/home/workspace/ByteTrack/detectors/yolov8m.pt')
        else:
            assert False, "detector model not in {}".format(model_sets)

    # def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
    #     from yolox.data import (
    #         VOCDetection,
    #         TrainTransform,
    #         YoloBatchSampler,
    #         DataLoader,
    #         InfiniteSampler,
    #         #MosaicDetection,
    #         #worker_init_reset_seed,
    #     )
    #     #from yolox.data.datasets.voc import VOCDataset
    #     from yolox.data.datasets.mosaicdetection import MosaicDetection
    #     #from yolox.utils import (
    #         #wait_for_the_master,
    #         #get_local_rank,
    #     #)
    #     #local_rank = get_local_rank()

    #     #with wait_for_the_master(local_rank):
    #     dataset = VOCDetection(
    #         data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
    #         image_sets=[('2012', 'trainval')],
    #         img_size=self.input_size,
    #         preproc=TrainTransform(
    #             rgb_means=(0.485, 0.456, 0.406),
    #             std=(0.229, 0.224, 0.225),
    #             max_labels=500,
    #         ),
    #     )

    #     dataset = MosaicDetection(
    #         dataset,
    #         mosaic=not no_aug,
    #         img_size=self.input_size,
    #         preproc=TrainTransform(
    #             rgb_means=(0.485, 0.456, 0.406),
    #             std=(0.229, 0.224, 0.225),
    #             max_labels=1000,
    #         ),
    #         degrees=self.degrees,
    #         translate=self.translate,
    #         scale=self.scale,
    #         shear=self.shear,
    #         perspective=self.perspective,
    #         enable_mixup=self.enable_mixup,
    #     )

    #     self.dataset = dataset

    #     if is_distributed:
    #         batch_size = batch_size // dist.get_world_size()

    #     sampler = InfiniteSampler(
    #         len(self.dataset), seed=self.seed if self.seed else 0
    #     )

    #     batch_sampler = YoloBatchSampler(
    #         sampler=sampler,
    #         batch_size=batch_size,
    #         drop_last=False,
    #         input_dimension=self.input_size,
    #         mosaic=not no_aug,
    #     )

    #     dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
    #     dataloader_kwargs["batch_sampler"] = batch_sampler

    #     # Make sure each process has different random seed, especially for 'fork' method
    #     #dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    #     train_loader = DataLoader(self.dataset, **dataloader_kwargs)

    #     return train_loader

    # def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
    #     from yolox.data import VOCDetection, ValTransform

    #     valdataset = VOCDetection(
    #         data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
    #         image_sets=[('2012', 'test')],
    #         img_size=self.test_size,
    #         preproc=ValTransform(
    #             rgb_means=(0.485, 0.456, 0.406),
    #             std=(0.229, 0.224, 0.225),
    #         ),
    #     )

    #     if is_distributed:
    #         batch_size = batch_size // dist.get_world_size()
    #         sampler = torch.utils.data.distributed.DistributedSampler(
    #             valdataset, shuffle=False
    #         )
    #     else:
    #         sampler = torch.utils.data.SequentialSampler(valdataset)

    #     dataloader_kwargs = {
    #         "num_workers": self.data_num_workers,
    #         "pin_memory": True,
    #         "sampler": sampler,
    #     }
    #     dataloader_kwargs["batch_size"] = batch_size
    #     val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    #     return val_loader

    # def get_evaluator(self, batch_size, is_distributed, testdev=False):
    #     from yolox.evaluators import VOCEvaluator

    #     val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
    #     evaluator = VOCEvaluator(
    #         dataloader=val_loader,
    #         img_size=self.test_size,
    #         confthre=self.test_conf,
    #         nmsthre=self.nmsthre,
    #         num_classes=self.num_classes,
    #         #testdev=testdev,
    #     )
    #     return evaluator
