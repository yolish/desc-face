import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH


class DescriptorHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3, descriptor_dim=64):
        super(DescriptorHead,self).__init__()
        self.descriptor_dim = descriptor_dim
        self.conv1x1 = nn.Conv2d(inchannels,descriptor_dim*num_anchors,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, self.descriptor_dim)


class DescRetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(DescRetinaFace, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.multi_scale_descriptor_head = cfg['ms_desciprtor']
        self.desc_head = self._make_desc_head(fpn_num=3, inchannels=cfg['out_channel'], descriptor_dim=cfg['descriptor_dim'])

        self.class_head = nn.Linear(cfg['descriptor_dim'], 2)
        self.bbox_head = nn.Linear(cfg['descriptor_dim'], 4)
        self.landmarks_head = nn.Linear(cfg['descriptor_dim'], 10)

        self.normalize = cfg.get('triplet_loss')

    def _make_desc_head(self,fpn_num=3,inchannels=64,anchor_num=2, descriptor_dim=64):
        if self.multi_scale_descriptor_head:
            desc_head = nn.ModuleList()
            for i in range(fpn_num):
                desc_head.append(DescriptorHead(inchannels,anchor_num,descriptor_dim))
        else:
            desc_head = DescriptorHead(inchannels,anchor_num,descriptor_dim)
        return desc_head

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        if self.multi_scale_descriptor_head:
            descriptors = torch.cat([self.desc_head[i](feature) for i, feature in enumerate(features)], dim=1)
        else:
            descriptors = torch.cat([self.desc_head(feature) for i, feature in enumerate(features)], dim=1)

        if self.normalize:
            descriptors = F.normalize(descriptors, p=2, dim=2)
        classifications = self.class_head(descriptors)
        ldm_regressions = self.landmarks_head(descriptors)
        bbox_regressions = self.bbox_head(descriptors)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions, descriptors)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output