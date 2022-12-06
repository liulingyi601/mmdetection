# Copyright (c) OpenMMLab. All rights reserved.
from tkinter.messagebox import NO
from turtle import forward
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32
from mmcv.runner.base_module import Sequential
from mmdet.core import (MlvlPointGenerator, bbox_overlaps, build_assigner,
                        build_prior_generator, build_sampler, multi_apply,anchor_inside_flags,unmap,images_to_levels,
                        reduce_mean)
from ..builder import HEADS, build_loss
from .atss_head import ATSSHead
from .fcos_head import FCOSHead
from mmcv.cnn.bricks.transformer import build_positional_encoding
import pdb
INF = 1e8



def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]



class cross_deformable_conv(nn.Module):
    def __init__(self, strides, in_channles,cdf_conv):
        super(cross_deformable_conv, self).__init__()
        self.strides=strides
        self.num_samples = cdf_conv.num_samples
        self.num_levels = len(strides)
        self.num_heads = cdf_conv.num_heads
        self.in_channles = in_channles
        self.use_pos = cdf_conv.use_pos
        self.num_head_channles = int(in_channles / self.num_heads)
        assert self.num_head_channles*self.num_heads == self.in_channles
        self.value_conv =  nn.Conv2d(in_channles, in_channles, 1)
        self.offset_conv = nn.Conv2d(in_channles, self.num_samples * self.num_heads * self.num_levels *2, 3, padding=1)
        self.weight_conv = nn.Conv2d(in_channles, self.num_samples * self.num_heads * self.num_levels, 3, padding=1)
        if self.use_pos:
            self.level_embeds = nn.Parameter(torch.Tensor(self.num_levels, self.in_channles))
        self.norm_conv = nn.Conv2d(in_channles, in_channles, 1)

        self.norm_layer1 = nn.LayerNorm(in_channles)
        self.FFN = nn.Sequential(
                    nn.Linear(in_channles, in_channles*2), 
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channles*2, in_channles))
        self.norm_layer2 = nn.LayerNorm(in_channles)

    def forward(self, feats, all_level_feat_points, positional_encodings=None):
        queries = []
        values=[]
        if self.use_pos:
            for i, feat in enumerate(feats): 
                values.append(self.value_conv(feat))
                queries.append(feat+positional_encodings[i]+self.level_embeds[i][None,:,None,None])
        else:
            for i, feat in enumerate(feats): 
                values.append(self.value_conv(feat))
                queries.append(feat)

        out_feats = []
        
        for i, query in enumerate(queries):
            N, _, H, W = query.size()
            points = (all_level_feat_points[i] / self.strides[i]).view(H, W, 2)
            offset = self.offset_conv(query)
            sample_weight = F.softmax(self.weight_conv(query).view(N*self.num_heads,1,self.num_levels*self.num_samples, H, W), dim=2)
            offset = offset.reshape(N*self.num_heads, self.num_levels, self.num_samples,2, H,W).permute(0,1,2,4,5,3).contiguous() #bs*num_heads, num_levels, num_samples, H_, W_, 2
            offset[...,0] = (offset[...,0] + points[None,None, None, :, :, 0]) / W
            offset[...,1] = (offset[...,1] + points[None,None, None, :, :, 1]) / H
            sample_location = offset * 2 - 1
            sample_feat = []
            for j in range(self.num_levels):
                sample_feat.append(F.grid_sample(values[j].view(N*self.num_heads,self.num_head_channles, values[j].shape[-2], values[j].shape[-1]), sample_location[:,j].flatten(2,3), mode='bilinear',padding_mode='zeros',align_corners=False))
            sample_feats = (torch.cat(sample_feat, dim=2).view(N*self.num_heads,self.num_head_channles, self.num_levels*self.num_samples,H, W) * sample_weight).sum(2).view(N, -1, H, W)
            out_feat = self.norm_conv(sample_feats) + feats[i]
            out_feat = out_feat.permute(0,2,3,1).contiguous()
            out_feat = self.norm_layer1(out_feat)
            out_feat = self.FFN(out_feat) + out_feat
            out_feat = self.norm_layer2(out_feat)
            out_feats.append(out_feat.permute(0, 3, 1, 2).contiguous())
        return out_feats


@HEADS.register_module()
class BGMSRefineHead(FCOSHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 cdf_conv=dict(num_heads=1, num_samples=5, use_pos=False),
                #  use_pos=False,
                 bbox_weight_cfg='pred',
                 use_refine_vfl=True,
                 sample_weight=False,
                 num_samples=5,
                #  cross_norm_cfg='ln',
                 auto_weighted_loss=False,
                 weight_clamp=True,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 reg_denoms=[32,64,128],
                 center_sampling=False,
                 center_sample_radius=1.5,
                 sync_num_pos=True,
                 gradient_mul=0.1,
                 bbox_norm_type='reg_denom',
                 loss_cls_fl=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 use_vfl=True,
                 loss_cls=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
                 loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                #  use_atss=True,
                 reg_decoded_bbox=True,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     ratios=[1.0],
                     octave_base_scale=8,
                     scales_per_octave=1,
                     center_offset=0.0,
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='vfnet_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        # dcn base offsets, adapted from reppoints_head.py
        self.weight_clamp=weight_clamp
        self.bbox_weight_cfg=bbox_weight_cfg
        self.cdf_conv=cdf_conv
        self.auto_weighted_loss = auto_weighted_loss
        self.sample_weight=sample_weight
        self.use_refine_vfl=use_refine_vfl
        self.num_samples=num_samples
        super(FCOSHead, self).__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.regress_ranges = regress_ranges
        self.reg_denoms = reg_denoms
        self.num_stage = len(self.strides)
        # self.reg_denoms[-1] = self.reg_denoms[-2] * 2
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.sync_num_pos = sync_num_pos
        self.bbox_norm_type = bbox_norm_type
        self.gradient_mul = gradient_mul
        self.use_vfl = use_vfl
        if self.use_vfl:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = build_loss(loss_cls_fl)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.anchor_center_offset = anchor_generator['center_offset']
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.atss_prior_generator = build_prior_generator(anchor_generator)
        self.fcos_prior_generator = MlvlPointGenerator(anchor_generator['strides'], self.anchor_center_offset)
        self.prior_generator = self.fcos_prior_generator
        if self.cdf_conv.use_pos:
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=self.feat_channels/2,
                normalize=True,
                offset=-0.5)
            self.positional_encoding = build_positional_encoding(positional_encoding)
    @property
    def num_anchors(self):
        """
        Returns:
            int: Number of anchors on each point of feature map.
        """
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
                      'please use "num_base_priors" instead')
        return self.num_base_priors

    @property
    def anchor_generator(self):
        warnings.warn('DeprecationWarning: anchor_generator is deprecated, '
                      'please use "atss_prior_generator" instead')
        return self.prior_generator

    def _init_layers(self):
        self.cls_layer = nn.ModuleList()
        self.reg_layer = nn.ModuleList()

        for i in range(self.stacked_convs):
            self.cls_layer.append(cross_deformable_conv(self.strides, self.feat_channels, self.cdf_conv))
            self.reg_layer.append(cross_deformable_conv(self.strides, self.feat_channels, self.cdf_conv))
        
        self.relu = nn.ReLU(inplace=True)
        self.vfnet_reg_conv = ConvModule(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.conv_bias)
        self.vfnet_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.reg_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1)
        self.cls_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.vfnet_reg_refine = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales_refine = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.vfnet_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.vfnet_reg_conv_weight = nn.Conv2d(self.feat_channels, self.num_samples * len(self.strides), 1)
        self.vfnet_cls_conv_weight = nn.Conv2d(self.feat_channels, self.num_samples * len(self.strides), 1)
        if self.auto_weighted_loss:
            self.auto_loss_weights = nn.Parameter(torch.ones(3))

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box iou-aware scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box offsets for each
                    scale level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                bbox_preds_refine (list[Tensor]): Refined Box offsets for
                    each scale level, each is a 4D-tensor, the channel
                    number is num_points * 4.
        """
        N = feats[0].size()[0]
        featmap_sizes = [feat.size()[-2:] for feat in feats]
        all_level_points = self.fcos_prior_generator.grid_priors(featmap_sizes, feats[0].dtype, feats[0].device)
        if self.cdf_conv.use_pos:
            positional_encodings = []
            for i, featmap_size in enumerate(featmap_sizes):
                mask = feats[0].new_zeros((N, featmap_size[0], featmap_size[1])).to(torch.bool)
                positional_encodings.append(self.positional_encoding(mask))
        else:
            positional_encodings=None


        for cls_layer in self.cls_layer:
            cls_feats = cls_layer(feats, all_level_points, positional_encodings)
        for reg_layer in self.reg_layer:
            reg_feats = reg_layer(feats, all_level_points, positional_encodings)
        cls_scores = []
        bbox_preds = []
        bbox_pred_refines = []
        for i, reg_feat in enumerate(reg_feats):
            N, _, W, H = reg_feat.size()
            reg_feat_init = self.vfnet_reg_conv(reg_feat)
            reg_feat_weight = F.softmax(self.vfnet_reg_conv_weight(reg_feat_init),dim=1)
            cls_feat_weight = F.softmax(self.vfnet_cls_conv_weight(reg_feat_init),dim=1)
            if self.bbox_norm_type == 'reg_denom':
                bbox_pred = self.scales[i](
                    self.vfnet_reg(reg_feat_init)).float().exp() * self.reg_denoms[i]
            elif self.bbox_norm_type == 'stride':
                bbox_pred = self.scales[i](
                    self.vfnet_reg(reg_feat_init)).float().exp() * self.strides[i]
            else:
                raise NotImplementedError
            # import pdb
            # pdb.set_trace()
            sample_location, cls_sample_location = self.gen_sample_location(bbox_pred, all_level_points[i], self.strides[i])
            sampling_reg_list = []
            sampling_cls_list = []
            for j in range(len(self.strides)):
                # pdb.set_trace()
                sampling_reg_list.append(F.grid_sample(reg_feats[j], sample_location, mode='bilinear',padding_mode='zeros',align_corners=False).view(N, -1, W, H, self.num_samples).permute(0,1,4,2,3).contiguous())
                sampling_cls_list.append(F.grid_sample(cls_feats[j], cls_sample_location, mode='bilinear',padding_mode='zeros',align_corners=False).view(N, -1, W, H, self.num_samples).permute(0,1,4,2,3).contiguous())
            # import pdb
            # pdb.set_trace()
            refine_reg_feat = (torch.cat(sampling_reg_list, dim=2) * reg_feat_weight[:,None,:,:,:]).sum(2)
            # cls_feat = (torch.cat(sampling_cls_list, dim=2) * reg_feat_weight[:,None,:,:,:]).sum(2)
            refine_cls_feat = (torch.cat(sampling_cls_list, dim=2) * cls_feat_weight[:,None,:,:,:]).sum(2)

            # add a conv 
            refine_reg_feat = self.relu(self.reg_conv(refine_reg_feat))
            refine_cls_feat = self.relu(self.cls_conv(refine_cls_feat))

            bbox_pred_refine = self.scales_refine[i](
                self.vfnet_reg_refine(refine_reg_feat)).float().exp()
            bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()
            cls_score = self.vfnet_cls(refine_cls_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            bbox_pred_refines.append(bbox_pred_refine)
        if self.training:
            return cls_scores, bbox_preds, bbox_pred_refines
        else:
            return cls_scores, bbox_pred_refines

    def gen_sample_location(self, bbox_pred, points, stride):
        """Compute the sample location.

        Args:
            bbox_pred (Tensor): Predicted bbox distance offsets (l, r, t, b).
            gradient_mul (float): Gradient multiplier.
            stride (int): The corresponding stride for feature maps,
                used to project the bbox onto the feature map.

        Returns:
            dcn_offsets (Tensor): The offsets for deformable convolution.
        """
        bbox_pred_grad_mul = (1 - self.gradient_mul) * bbox_pred.detach() + \
            self.gradient_mul * bbox_pred
        # map to the feature map scale
        N, C, H, W = bbox_pred.size()

        x1 = bbox_pred_grad_mul[:, 0, :, :]
        y1 = bbox_pred_grad_mul[:, 1, :, :]
        x2 = bbox_pred_grad_mul[:, 2, :, :]
        y2 = bbox_pred_grad_mul[:, 3, :, :]

        sample_offset = bbox_pred.new_zeros(
            N,  H, W, self.num_samples, 2)
        sample_offset[:, :, :, 0, 0] = -1.0 * y1  # -y1
        sample_offset[:, :, :, 1, 1] = -1.0 * x1
        sample_offset[:, :, :, 3, 1] = x2  # -x1
        sample_offset[:, :, :, 4, 0] = y2  # x2
    
        cls_offset = bbox_pred.new_zeros(
            N,  H, W, self.num_samples, 2)
        cls_offset[:, :, :, 0, 0] = -1.0 * y1 / 2
        cls_offset[:, :, :, 0, 1] = -1.0 * x1 / 2
        cls_offset[:, :, :, 1, 0] = -1.0 * y1 / 2
        cls_offset[:, :, :, 1, 1] = x2 / 2
        cls_offset[:, :, :, 3, 0] = y2 / 2
        cls_offset[:, :, :, 3, 1] = x2 / 2
        cls_offset[:, :, :, 3, 0] = y2 / 2
        cls_offset[:, :, :, 3, 1] = -1.0 * x1 / 2

        points = points.view(H,W,2)[None,:,:,None]

        sample_location = sample_offset + points[:,:,:,:,:2]
        sample_location[:,:,:,:,0] = sample_location[:,:,:,:,0] / (W * stride)
        sample_location[:,:,:,:,1] = sample_location[:,:,:,:,1] / (H * stride)
        sample_location = sample_location.view(N, W, H* self.num_samples, 2) * 2 - 1

        cls_sample_location = cls_offset + points[:,:,:,:,:2]
        cls_sample_location[:,:,:,:,0] = cls_sample_location[:,:,:,:,0] / (W * stride)
        cls_sample_location[:,:,:,:,1] = cls_sample_location[:,:,:,:,1] / (W * stride)
        cls_sample_location = cls_sample_location.view(N, W, H* self.num_samples, 2) * 2 - 1


        return sample_location, cls_sample_location

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bbox_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box offsets for each
                scale level, each is a 4D-tensor, the channel number is
                num_points * 4.
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level, each is a 4D-tensor, the channel
                number is num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # list（n_levels * tensor[N,C,W,H]) >>> list(N*list(n_levels*tensor[W*H, C])) >>> list(N*tensor[n_anchor, C])
        bbox_preds_list = levels_to_images(bbox_preds)
        bbox_preds_refine_list = levels_to_images(bbox_preds_refine)
        cls_scores_list = levels_to_images(cls_scores)

        all_labels, all_label_weights, pos_bbox_targets, pos_pre_boxes, pos_pre_boxes_refine, pos_bbox_weights, pos_bbox_weights_refine = self.get_targets(
            bbox_preds_list, bbox_preds_refine_list, featmap_sizes, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
        # labels:list(N*tensor[n_anchors, n_classes])
        # label_weights:list(N*tensor[n_anchors])
        # bbox_targets,pre_boxes,pre_boxes_refine:list(N*tensor(n_pos, 4))
        # bbox_weights, bbox_weights_refine:list(N*tensor(n_pos))

        flatten_cls_scores=torch.cat(cls_scores_list) # tensor(N*n_anchors, n_classes)
        flatten_labels = torch.cat(all_labels) # tensor(N*n_anchors, n_classes)
        flatten_label_weights=torch.cat(all_label_weights) # tensor(N*n_anchors)

        flatten_bbox_targets=torch.cat(pos_bbox_targets)# tensor(n_pos, 4)
        flatten_pre_boxes=torch.cat(pos_pre_boxes) # tensor(n_pos, 4)
        flatten_pre_boxes_refine=torch.cat(pos_pre_boxes_refine) # tensor(n_pos, 4)
        flatten_bbox_weights=torch.cat(pos_bbox_weights) # tensor(n_pos)
        flatten_bbox_weights_refine=torch.cat(pos_bbox_weights_refine) # tensor(n_pos)
        num_pos=(flatten_labels>0).sum()
        # sync num_pos across all gpus
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                flatten_labels.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos
        if self.weight_clamp:
            bbox_avg_factor_ini = reduce_mean(
                flatten_bbox_weights.sum()).clamp_(min=1).item()
            bbox_avg_factor_rf = reduce_mean(
                flatten_bbox_weights_refine.sum()).clamp_(min=1).item()
        else:
            bbox_avg_factor_ini = reduce_mean(
                flatten_bbox_weights.sum()).item()
            bbox_avg_factor_rf = reduce_mean(
                flatten_bbox_weights_refine.sum()).item()
        # pdb.set_trace()
        # pos_decoded_bbox_preds, pos_decoded_target_preds, pos_decoded_bbox_preds_refine: tensor[n_pos, 4]
        # bbox_weights_ini, bbox_weights_rf: tensor[n_pos]
        # flatten_cls_scores, flatten_labels: tensor[N*n_anchors, n_classes]
        if num_pos > 0:
            loss_bbox = self.loss_bbox(
                flatten_pre_boxes,
                flatten_bbox_targets.detach(),
                weight=flatten_bbox_weights.detach(),
                avg_factor=bbox_avg_factor_ini)

            loss_bbox_refine = self.loss_bbox_refine(
                flatten_pre_boxes_refine,
                flatten_bbox_targets.detach(),
                weight=flatten_bbox_weights_refine.detach(),
                avg_factor=bbox_avg_factor_rf)
        else:
            loss_bbox = flatten_pre_boxes.sum() * 0
            loss_bbox_refine = flatten_pre_boxes_refine.sum() * 0


        if self.use_vfl:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels.detach(),
                avg_factor=num_pos_avg_per_gpu)
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels.detach(),
                weight=flatten_label_weights.detach(),
                avg_factor=num_pos_avg_per_gpu)
        # pdb.set_trace()
        if self.auto_weighted_loss:
            loss_cls_aw = (-self.auto_loss_weights[0]).exp()*loss_cls + self.auto_loss_weights[0]
            loss_bbox_refine_aw = (-self.auto_loss_weights[1]).exp()*loss_bbox_refine + self.auto_loss_weights[1]
            loss_bbox_aw = (-self.auto_loss_weights[2]).exp()*loss_bbox + self.auto_loss_weights[2]
            return dict(
                ori_cls=loss_cls,
                ori_bbox_refine=loss_bbox_refine,
                ori_bbox=loss_bbox,
                loss_cls=loss_cls_aw,
                loss_bbox=loss_bbox_aw,
                loss_bbox_rf=loss_bbox_refine_aw)
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_bbox_rf=loss_bbox_refine)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.atss_prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.atss_prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device=device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list
    def get_targets(self,
                    bbox_preds_list, 
                    bbox_preds_refine_list,
                    featmap_sizes,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """A wrapper for computing ATSS targets for points in multiple images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4). Default: None.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor): Bbox weights of all levels.
        """
        assert len(featmap_sizes) == self.atss_prior_generator.num_levels == self.fcos_prior_generator.num_levels
        num_imgs = len(img_metas)

        device = bbox_preds_list[0].device
        all_level_points = self.fcos_prior_generator.grid_priors(
            featmap_sizes, bbox_preds_list[0].dtype, device)
        points_list = [torch.cat(all_level_points)]*num_imgs

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, pos_bbox_targets, pos_pre_boxes, pos_pre_boxes_refine,
        pos_bbox_weights, pos_bbox_weights_refine) = multi_apply(
             self._get_target_single,
             bbox_preds_list,
             bbox_preds_refine_list,
             points_list,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas)
        if any([labels is None for labels in all_labels]):
            return None

        # labels:list(N*tensor[n_anchors, n_classes])
        # label_weights:list(N*tensor[n_anchors])
        # bbox_targets,pre_boxes,pre_boxes_refine:list(N*tensor(n_pos, 4))
        # bbox_weights, bbox_weights_refine:list(N*tensor(n_pos))

        return all_labels, all_label_weights, pos_bbox_targets, pos_pre_boxes, pos_pre_boxes_refine, pos_bbox_weights, pos_bbox_weights_refine

    def _get_target_single(self,
                           flat_bbox_preds,
                           flat_bbox_preds_refine,
                           flat_points,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        bbox_preds = flat_bbox_preds[inside_flags,:]
        bbox_preds_refine = flat_bbox_preds_refine[inside_flags,:]
        points = flat_points[inside_flags,:]




        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        if self.train_cfg.assigner.type=='UniformAssigner':
            decoder_bbox_preds_refine = self.bbox_coder.decode(points, bbox_preds_refine)
            assign_result = self.assigner.assign(decoder_bbox_preds_refine, anchors, gt_bboxes, gt_bboxes_ignore, gt_labels)
        else:
            assign_result = self.assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds
        num_pos = len(pos_inds)

        pos_bbox_targets = anchors.new_full((num_pos, 4), 0, dtype=torch.float)
        pos_pre_boxes = anchors.new_full((num_pos, 4), 0, dtype=torch.float)
        pos_pre_boxes_refine = anchors.new_full((num_pos, 4), 0, dtype=torch.float)
        pos_bbox_weights = anchors.new_full((num_pos, ), 0, dtype=torch.float)
        pos_bbox_weights_refine = anchors.new_full((num_pos, ), 0, dtype=torch.float)

        all_labels = anchors.new_full((num_valid_anchors, self.num_classes), 0, dtype=torch.float)
        all_label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)



        # pdb.set_trace()
        if num_pos > 0:
            pos_bbox_targets[:, :] = sampling_result.pos_gt_bboxes
            pos_pre_boxes[:,:] = self.bbox_coder.decode(points[pos_inds], bbox_preds[pos_inds])
            pos_pre_boxes_refine[:,:] = self.bbox_coder.decode(points[pos_inds], bbox_preds_refine[pos_inds])

            for i in range(sampling_result.num_gts):
                sample_pos_ids = pos_gt_inds==i
                sample_inds = pos_inds[sample_pos_ids]
                if self.bbox_weight_cfg=='anchor':
                    sample_weights = assign_result.max_overlaps[sample_inds]
                    sample_weights_refine = assign_result.max_overlaps[sample_inds]
                elif self.bbox_weight_cfg=='pred':
                    sample_bbox_preds = pos_pre_boxes[sample_pos_ids]
                    sample_bbox_preds_refine = pos_pre_boxes_refine[sample_pos_ids]
                    sample_bbox_targets = pos_bbox_targets[sample_pos_ids]
                    sample_weights = bbox_overlaps(sample_bbox_preds,sample_bbox_targets,is_aligned=True).clamp(min=1e-6)
                    sample_weights_refine = bbox_overlaps(sample_bbox_preds_refine,sample_bbox_targets,is_aligned=True).clamp(min=1e-6)
                elif self.bbox_weight_cfg=='cons':
                    sample_weights=anchors.new_full((len(sample_inds), ), 1, dtype=torch.float)
                    sample_weights_refine = anchors.new_full((len(sample_inds), ), 1, dtype=torch.float)
                if self.sample_weight==True:
                    avg_factor = sample_weights.sum()
                    avg_factor_refine = sample_weights_refine.sum()

                else:
                    avg_factor=1
                    avg_factor_refine=1
                pos_bbox_weights[sample_pos_ids] = sample_weights / avg_factor
                pos_bbox_weights_refine[sample_pos_ids] = sample_weights_refine /avg_factor_refine
                if self.use_vfl:
                    if self.use_refine_vfl:
                        all_labels[sample_inds,gt_labels[i]]=sample_weights_refine
                    else:
                        all_labels[sample_inds,gt_labels[i]]=sample_weights

                else:
                    all_labels[sample_inds,gt_labels[i]]=1

            if self.train_cfg.pos_weight <= 0:
                all_label_weights[pos_inds] = 1.0
            else:
                all_label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            all_label_weights[neg_inds] = 1.0
        # pdb.set_trace()
        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            all_labels = unmap(all_labels, num_total_anchors, inside_flags)
            all_label_weights = unmap(all_label_weights, num_total_anchors, inside_flags)
        # labels:tensor[n_anchors, n_classes]
        # label_weights:tensor[n_anchors]
        # bbox_targets,pre_boxes,pre_boxes_refine:tensor(n_pos, 4)
        # bbox_weights, bbox_weights_refine:tensor(n_pos)
        return (all_labels, all_label_weights, pos_bbox_targets, pos_pre_boxes,pos_pre_boxes_refine, pos_bbox_weights,pos_bbox_weights_refine)

    def transform_bbox_targets(self, decoded_bboxes, mlvl_points, num_imgs):
        """Transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format.

        Args:
            decoded_bboxes (list[Tensor]): Regression targets of each level,
                in the form of (x1, y1, x2, y2).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            num_imgs (int): the number of images in a batch.

        Returns:
            bbox_targets (list[Tensor]): Regression targets of each level in
                the form of (l, t, r, b).
        """
        # TODO: Re-implemented in Class PointCoder
        assert len(decoded_bboxes) == len(mlvl_points)
        num_levels = len(decoded_bboxes)
        mlvl_points = [points.repeat(num_imgs, 1) for points in mlvl_points]
        bbox_targets = []
        for i in range(num_levels):
            bbox_target = self.bbox_coder.encode(mlvl_points[i],
                                                 decoded_bboxes[i])
            bbox_targets.append(bbox_target)

        return bbox_targets
    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override the method in the parent class to avoid changing para's
        name."""
        pass

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """

        warnings.warn(
            '`_get_points_single` in `VFNetHead` will be '
            'deprecated soon, we support a multi level point generator now'
            'you can get points of a single level feature map'
            'with `self.fcos_prior_generator.single_level_grid_priors` ')

        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        # to be compatible with anchor points in ATSS
        points = torch.stack(
                (x.reshape(-1), y.reshape(-1)), dim=-1) + \
                     stride * self.anchor_center_offset

        return points

