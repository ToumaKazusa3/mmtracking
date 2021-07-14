_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot15-private-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        'work_dirs/publish_model/faster-rcnn_r50_fpn_4e_mot15-9e00ac7f.pth'  # noqa: E501
    ))
data_root = 'data/MOT15/'
test_set = 'train'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    val=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    test=dict(
        ann_file=data_root + f'annotations/{test_set}_cocoformat.json',
        img_prefix=data_root + test_set))
