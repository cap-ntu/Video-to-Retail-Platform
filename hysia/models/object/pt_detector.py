# @Time    : 2/11/18 2:17 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : pt_detector.py

import mmcv
import os.path as osp
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from .pt_inference import inference_detector

from utils.logger import Logger

logger = Logger(
    name="pytorch_object_detector",
    severity_levels={"StreamHandler": "INFO"}
)


class PT_MMdetector(object):
    """
    This is pytorch version detector.
    The third code comes from https://github.com/open-mmlab/mmdetection
    """

    def __init__(self, model_path, device, cfg_path='configs/faster_rcnn_features.py'):
        '''

        :param cfg_path:
        :param model_path:
        :param device:
        '''
        self._cfg = mmcv.Config.fromfile(cfg_path)
        self._cfg.model.pretrained = None
        self._whole_handle = None
        self._roi_handle = None

        self._device = device

        self.model = build_detector(self._cfg.model, test_cfg=self._cfg.test_cfg)

        if not osp.isfile(model_path):
            raise IOError('{} is not a checkpoint file. \n Please run the download_test_models.py'.format(model_path))
        _ = load_checkpoint(self.model, model_path)

        self.extract_feature_maps()
        self.extract_roi_features()

        logger.info('Finish loading PyTorch Faster RCNN detector and feature extractor')

    @property
    def cfg(self):
        return self._cfg

    def hook(self, module, input, output):
        print(module)
        print(output.size())

    def remove_hook(self):
        if self._whole_handle is not None:
            self._whole_handle.remove()
        if self._roi_handle is not None:
            self._roi_handle.remove()

    def extract_feature_maps(self):
        self._whole_handle = self.model.backbone.layer4.register_forward_hook(self.hook)

    def extract_roi_features(self):
        self._roi_handle = self.model.bbox_roi_extractor.register_forward_hook(self.hook)

    def detect(self, img):
        """

        :param img:
        :return:
        """
        result = inference_detector(self.model, img, self._cfg, self._device)

        self.remove_hook()

        return result
