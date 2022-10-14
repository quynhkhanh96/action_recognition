import torch
import torch.nn as nn
import torch.nn.functional as F

class Recognizer3D(nn.Module):
    def __init__(self, backbone, cls_head=None):
        super().__init__()        
        self.backbone = backbone

        self.cls_head = cls_head if cls_head else None
        # max_testing_views should be int
        self.max_testing_views = None
        self.feature_extraction = False

        # mini-batch blending, e.g. mixup, cutmix, etc.
        self.blending = None
        self.init_weights()
        self.fp16_enabled = False


    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None
    
    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()
        if self.with_neck:
            self.neck.init_weights()
    
    # @auto_fp16()
    def extract_feat(self, imgs):
        if (hasattr(self.backbone, 'features')
                and self.backbone_from == 'torchvision'):
            x = self.backbone.features(imgs)
        elif self.backbone_from == 'timm':
            x = self.backbone.forward_features(imgs)
        elif self.backbone_from == 'mmcls':
            x = self.backbone(imgs)
            if isinstance(x, tuple):
                assert len(x) == 1
                x = x[0]
        else:
            x = self.backbone(imgs)
        return x
    
    def average_clip(self, cls_score, num_segs=1):
        if 'average_clips' not in self.test_cfg.keys():
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips is None:
            return cls_score

        batch_size = cls_score.shape[0]
        cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=1)

        return cls_score

    def forward(self, imgs):
        # use mmaction2's `Recognizer3D.forward_train()`, but without `loss_cls = self.cls_head.loss(...)`,
        # just return cls_score
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)

        return cls_score

    # Just it, don't use `train_strep()` and `val_step()` of mmaction2's `BaseRecognizer`