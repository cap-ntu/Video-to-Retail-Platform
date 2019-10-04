# @Time    : 7/11/18 5:48 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : faster_rcnn_features.py

from .two_stage import TwoStageDetector


class FasterRCNN_Features(TwoStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(FasterRCNN_Features, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    # def extract_feat(self, img):
    #     x = self.backbone(img)
    #     for i in x:
    #         print(i.size())
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x
