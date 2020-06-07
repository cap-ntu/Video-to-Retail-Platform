import ssl
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList

from common.engine import BaseEngine
from .mask_rcnn_predictor import COCODemo

# cancel ssl certificate verify
ssl._create_default_https_context = ssl._create_unverified_context


class Engine(BaseEngine):
    CFG_ROOT = Path(__file__).parent.absolute() / 'third/maskrcnn-benchmark/configs'

    def __init__(self, config):
        super().__init__(config)
        self._load_model(self.config)

    def _load_model(self, model_name: str):
        self._model_name = model_name
        self._config = self._load_cfg()
        self._model = COCODemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
        )

    def reset_model_version(self, model_name: str):
        self._load_model(model_name)

    def _load_cfg(self):
        model_path = Path(self._model_name).with_suffix('.yaml')
        full_path = self.CFG_ROOT / model_path
        print('loading configuration from {}'.format(full_path))
        cfg.merge_from_file(full_path)
        return cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    @staticmethod
    def decode_bbox(predictions: BoxList):
        """
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        Returns:
            label, boxes, scores (list, list, list): a tuple containing list of
                labels, boxes and scores.
        """
        # get label
        label_ids = predictions.get_field('labels').tolist()
        boxes = predictions.bbox
        boxes = boxes.to(torch.int64).tolist()
        scores = predictions.get_field('scores').tolist()

        if predictions.has_field('mask'):
            mask = predictions.get_field('mask').tolist()
        else:
            mask = None

        return label_ids, boxes, scores, mask

    def single_predict(self, np_array: np.ndarray, **kwargs) -> Dict[str, list]:
        height, width, _ = np_array.shape

        predictions = self._model.compute_prediction(np_array)
        top_predictions = self._model.select_top_predictions(predictions)
        label_ids, boxes, scores, mask = self.decode_bbox(top_predictions)
        labels = [self._model.CATEGORIES[i] for i in label_ids]

        return {
            'labels': labels,
            'label_ids': label_ids,
            'boxes': boxes,
            'scores': scores,
            'mask': mask,
            'width': width,
            'height': height
        }

    def batch_predict(self, *args, **kwargs):
        print('Hello world from batch predict.')
