import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes, pretrained=True, min_size=480, max_size=800):
    """Build a Faster R-CNN with a fresh box predictor sized for our classes.

    ``num_classes`` counts the foreground classes only; the predictor gets one
    extra slot for torchvision's implicit background class at index 0.

    ``min_size``/``max_size`` size the model's internal GeneralizedRCNNTransform.
    These live in ``TrainConfig`` but were never passed through, so the model
    silently used torchvision's 800/1333 defaults — noticeably more VRAM than
    the config asks for. Training and inference must use the SAME values or
    detections shift.

    ``pretrained=False`` skips BOTH the COCO detection weights and the ImageNet
    backbone weights. torchvision defaults ``weights_backbone`` to
    IMAGENET1K_V1 even when ``weights=None``, which downloads ~98 MB that a
    subsequent ``load_state_dict`` immediately overwrites — a pure cold-start
    tax on inference. Pass it explicitly as None.
    """
    if pretrained:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            min_size=min_size,
            max_size=max_size,
            rpn_pre_nms_top_n_train=2000,
            rpn_post_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_test=1000,
        )
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size,
            rpn_pre_nms_top_n_train=2000,
            rpn_post_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_test=1000,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model
