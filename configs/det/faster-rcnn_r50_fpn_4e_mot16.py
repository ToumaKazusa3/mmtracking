USE_MMDET = True
_base_ = ['./faster-rcnn_r50_fpn_4e_mot16-half.py']
data_root = 'data/MOT16/'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    val=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    test=dict(ann_file=data_root + 'annotations/train_cocoformat.json'))