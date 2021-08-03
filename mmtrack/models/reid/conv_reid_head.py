import math

import torch
import torch.functional as F
import torch.nn as nn
from mmcls.models.builder import HEADS
from mmcls.models.heads.base_head import BaseHead
from mmcv.cnn import normal_init
from mmdet.models.builder import build_loss
from mmdet.models.utils.gaussian_target import transpose_and_gather_feat


@HEADS.register_module()
class ConvReIDHead(BaseHead):
    """Conv head for re-identification."""

    def __init__(self,
                 in_channel,
                 feat_channel,
                 out_channel,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 max_objs=100,
                 num_classes=None):
        super(ConvReIDHead, self).__init__()
        assert num_classes
        self.in_channel = in_channel
        self.feat_channel = feat_channel
        self.out_channel = out_channel
        self.max_objs = max_objs
        self.num_classes = num_classes
        self.max_objs = max_objs
        self.emb_scale = math.sqrt(2) * math.log(self.num_classes - 1)
        self.reid_head = self._build_head()
        self.classifier = self._build_classifier()
        self.loss = build_loss(loss)

    def _build_head(self):
        """Build head for reid branch."""
        layer = nn.Sequential(
            nn.Conv2d(
                self.in_channel, self.feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channel, self.out_channel, kernel_size=1))
        return layer

    def _build_classifier(self):
        """Build classifier for reid branch."""
        classifier = nn.Linear(self.out_channels, self.num_classes)
        return classifier

    def init_weights(self):
        """Initalize model weights."""
        for m in self.reid_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.classifier, mean=0, std=0.01, bias=0)

    def get_gt_ind(self, gt_bboxes, feat_shape, img_shape):
        """Compute indices of targets in multiple images."""
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        ind = gt_bboxes[-1].new_zeros([bs, self.max_objs])
        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ind[batch_id][j] = cty_int * feat_w + ctx_int

        return ind

    def simple_test(self, x, batch_index):
        """Test without augmentation."""
        x = F.normalize(x, dim=1)
        id_feats = transpose_and_gather_feat(x, batch_index)
        return id_feats

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels, **kwargs):
        """Model forward."""
        batch_index = self.get_gt_ind(gt_bboxes, x.shape,
                                      img_metas[0]['pad_shape'])
        id_head = transpose_and_gather_feat(x, batch_index)
        id_head = self.emb_scale * F.normalize(id_head)
        id_output = self.classifier(id_head).contiguous()
        id_loss = self.loss(id_output, gt_labels)

        return id_loss
