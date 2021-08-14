import warnings

import torch
from mmdet.core import bbox2result
from mmdet.models import build_detector
from mmdet.models.utils.gaussian_target import (get_local_maximum,
                                                get_topk_from_heatmap)
from torch import nn
import mmcv

from mmtrack.core import track2result
from ..builder import MODELS, build_motion, build_reid, build_tracker
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class FairMOT(BaseMultiObjectTracker):
    """Simple online and realtime tracking with a deep association metric.

    Details can be found at `DeepSORT<https://arxiv.org/abs/1703.07402>`_.
    """

    def __init__(self,
                 detector=None,
                 reid=None,
                 tracker=None,
                 motion=None,
                 pretrains=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            if detector:
                detector_pretrain = pretrains.get('detector', None)
                if detector_pretrain:
                    detector.init_cfg = dict(
                        type='Pretrained', checkpoint=detector_pretrain)
                else:
                    detector.init_cfg = None
            if reid:
                reid_pretrain = pretrains.get('reid', None)
                if reid_pretrain:
                    reid.init_cfg = dict(
                        type='Pretrained', checkpoint=reid_pretrain)
                else:
                    reid.init_cfg = None

        if detector is not None:
            self.detector = build_detector(detector)

        if reid is not None:
            self.reid = build_reid(reid)

        if motion is not None:
            self.motion = build_motion(motion)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def get_inds(self, center_heatmap_pred, k=100, kernel=3):
        """Get batch index of heat points."""
        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)
        batch_scores, batch_index, *_ = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        return batch_index

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_instance_ids):
        """Forward function during training."""
        x = self.detector.extract_feat(img)

        losses = dict()
        det_loss = self.detector.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels)
        for key in det_loss:
            det_loss[key] = torch.exp(-self.s_det) * det_loss[key] * 0.5
        losses.update(det_loss)

        reid_loss = self.reid.forward_train(x[0], img_metas, gt_bboxes,
                                            gt_instance_ids)
        for key in reid_loss:
            reid_loss[key] = torch.exp(-self.s_id) * reid_loss[key] * 0.5
        losses.update(reid_loss)

        loss = {'loss_parm': (self.s_det + self.s_id) * 0.5}
        losses.update(loss)

        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    rescale=False,
                    public_bboxes=None,
                    **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.
            public_bboxes (list[Tensor], optional): Public bounding boxes from
                the benchmark. Defaults to None.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = tuple(img.size()[-2:])

        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)
        if hasattr(self.detector, 'roi_head'):
            # TODO: check whether this is the case
            if public_bboxes is not None:
                public_bboxes = [_[0] for _ in public_bboxes]
                proposals = public_bboxes
            else:
                proposals = self.detector.rpn_head.simple_test_rpn(
                    x, img_metas)
            det_bboxes, det_labels = self.detector.roi_head.simple_test_bboxes(
                x,
                img_metas,
                proposals,
                self.detector.roi_head.test_cfg,
                rescale=rescale)
            # TODO: support batch inference
            det_bboxes = det_bboxes[0]
            det_labels = det_labels[0]
            num_classes = self.detector.roi_head.bbox_head.num_classes
        elif hasattr(self.detector, 'bbox_head'):
            outs = self.detector.bbox_head(x)
            batch_index = self.get_inds(outs[0][0])
            result_list = self.detector.bbox_head.get_bboxes(
                *outs, img_metas=img_metas, rescale=rescale)
            # TODO: support batch inference
            det_bboxes = result_list[0][0]
            det_labels = result_list[0][1]
            num_classes = self.detector.bbox_head.num_classes
            # np_det_bboxes = det_bboxes.cpu().numpy()
            # mmcv.imshow_bboxes(img_meta['filename'], np_det_bboxes[np_det_bboxes[:, -1] > 0.1])
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        bboxes, labels, ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            feats=x[0],
            batch_index=batch_index,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)

        track_result = track2result(bboxes, labels, ids, num_classes)
        bbox_result = bbox2result(det_bboxes, det_labels, num_classes)
        return dict(bbox_results=bbox_result, track_results=track_result)
