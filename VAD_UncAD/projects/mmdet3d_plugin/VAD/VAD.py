import time
import copy
import torch.nn.functional as F
import numpy as np
import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from skimage.draw import polygon
import cv2
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric
from torch.distributions.laplace import Laplace
from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import VADCustomNuScenesDataset


@DETECTORS.register_module()
class VAD(MVXTwoStageDetector):
    """VAD model.
    """
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 fut_mode=6
                 ):

        super(VAD,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head['valid_fut_ts']

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.planning_metric = None
        # add nuscene map
        H, W=4.084, 1.85
        test_pipeline = []
        self.nuscene_map=VADCustomNuScenesDataset(ann_file="data_processed/vad/vad_nuscenes_infos_temporal_val.pkl", \
                                 data_root="data/nuscenes",pipeline=test_pipeline)
        self.pts = np.array([
            [-H / 2. + 0.5, W / 2.],
            [H / 2. + 0.5, W / 2.],
            [H / 2. + 0.5, -W / 2.],
            [-H / 2. + 0.5, -W / 2.],
        ])
        self.pts[:,[0,1]]=self.pts[:,[1,0]]
        self.sampleidx2index={}
        for index in range(len(self.nuscene_map)):
            sample_idx=self.nuscene_map.get_data_info(index)['sample_idx']
            self.sampleidx2index[sample_idx]=index


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,                          
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          prev_bev=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,   #这里的outs是指vad_head的forward得到的结果
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels
        ]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape #[1, 2, 6, 3, 384, 640]
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(  #[1, 10000, 256]
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas) #[1, 6, 256, 12, 20]
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore, prev_bev,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels)

        losses.update(losses_pts)
        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info['prev_bev'],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Test function"""
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]

        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = []
        for i, (bboxes, scores, labels, trajs, trajs_cls_scores, map_bboxes, \
                map_scores, map_labels, map_pts, map_betas) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result['trajs_3d'] = trajs.cpu()
            bbox_result['trajs_cls_scores_3d'] = trajs_cls_scores.cpu()
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts, map_betas)
            bbox_result.update(map_bbox_result)
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()
            bbox_result['ego_traj_cls_scores'] = outs['ego_traj_cls_scores'][i].cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_result['ego_fut_gt'] = ego_fut_trajs.cpu()
            bbox_result['ego_his_gt'] = ego_his_trajs.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        score_threshold = 0.4
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)

            bbox_result = c_bbox_results[0]
            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])
            # filter pred bbox by score_threshold
            mask = bbox_result['scores_3d'] > score_threshold       
            bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
            bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
            bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
            bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]
            bbox_result['trajs_cls_scores_3d'] = bbox_result['trajs_cls_scores_3d'][mask]

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'

            map_det_bbox = map_bbox_result['map_boxes_3d'].to(ego_fut_trajs.device)
            map_det_label = map_bbox_result['map_labels_3d'].to(ego_fut_trajs.device)
            map_det_scores = map_bbox_result['map_scores_3d'].to(ego_fut_trajs.device)
            map_det_pts = map_bbox_result['map_pts_3d'].to(ego_fut_trajs.device)
            map_det_betas = map_bbox_result['map_betas_3d'].to(ego_fut_trajs.device)

            det_bbox = bbox_result['boxes_3d'].tensor.to(ego_fut_trajs.device)
            det_label = bbox_result['labels_3d'].to(ego_fut_trajs.device)
            det_scores = bbox_result['scores_3d'].to(ego_fut_trajs.device)

            motion_traj_pred = bbox_result['trajs_3d'].to(ego_fut_trajs.device)
            motion_traj_scores = bbox_result['trajs_cls_scores_3d'].to(ego_fut_trajs.device)
            motion_traj_scores = motion_traj_scores.sigmoid()
            ego_fut_preds = bbox_result['ego_fut_preds'].to(ego_fut_trajs.device)
            ego_traj_cls_scores = bbox_result['ego_traj_cls_scores'].to(ego_fut_trajs.device)

            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0].to(ego_fut_trajs.device)
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]

            ego_fut_preds = ego_fut_preds.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            update_ego_traj_cls_scores, min_nll_per_mode,best_ego_traj_cls_scores, best_ego_fut_pred = self.select(map_det_pts,map_det_betas,det_bbox,
                                            det_label,
                                            det_scores,
                                            motion_traj_pred,
                                            motion_traj_scores, 
                                            ego_fut_preds, 
                                            ego_traj_cls_scores,
                                            ego_fut_cmd_idx, 
                                            )

            bbox_results[0]['plan_cls'] = update_ego_traj_cls_scores.cpu()

            matched_bbox_result = self.assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label)

            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result,
                matched_bbox_result, mapped_class_names) 

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                # add drivable area test
                img_metas[0]['sample_idx'],
                pred_ego_fut_trajs = best_ego_fut_pred,
                pred_ego_trajs_cls_scores = best_ego_traj_cls_scores,
                all_pred_ego_fut_trajs = ego_fut_pred[None],
                all_pred_ego_trajs_cls_scores = update_ego_traj_cls_scores,
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)

        return outs['bev_embed'], bbox_results, metric_dict

    def select(
        self,
        map_det_pts,
        map_det_betas,
        map_det_label,
        map_det_scores,
        det_bbox,
        det_label,
        det_scores,
        motion_pred,
        motion_cls,
        plan_preds,
        plan_cls,
        plan_cmd,

    ):


        det_bbox = det_bbox[None]
        det_scores = det_scores[None]
        motion_pred = motion_pred[None]
        motion_cls = motion_cls[None]
        plan_preds = plan_preds[None]
        plan_cls = plan_cls[None]

        det_confidence = det_scores 

        bs = 1  #only support batch_size=1 for testing
        bs_indices = torch.arange(bs, device=motion_cls.device)
        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, plan_cmd]
        plan_preds = plan_preds[bs_indices, plan_cmd]


        if det_bbox.shape[1] == 0:

            plan_cls,min_nll_per_mode = self.rescore_map_only(
                plan_cls,
                plan_preds,
                map_det_pts,
                map_det_betas,
                map_det_label,
                map_det_scores,
            )
            plan_cls = F.softmax(plan_cls, dim=-1)
            mode_idx = plan_cls.argmax(dim=-1)
            final_planning_score = plan_cls[bs_indices, mode_idx] 
            final_planning = plan_preds[bs_indices, mode_idx]
            return plan_cls, min_nll_per_mode, final_planning_score, final_planning

        plan_cls,min_nll_per_mode = self.rescore(
            plan_cls,
            plan_preds, 
            map_det_pts,
            map_det_betas,
            map_det_label,
            map_det_scores,
            motion_cls,
            motion_pred, 
            det_bbox,
            det_confidence,
        )

        plan_cls = F.softmax(plan_cls, dim=-1) 
        mode_idx = plan_cls.argmax(dim=-1)
        final_planning_score = plan_cls[bs_indices, mode_idx]
        final_planning = plan_preds[bs_indices, mode_idx]
        return plan_cls, min_nll_per_mode, final_planning_score, final_planning

    def rescore_map_only(
        self,
        plan_cls,
        plan_preds,
        map_det_pts,
        map_det_betas,
        map_det_label,
        map_det_scores,
        map_score_thresh=0.4,
        dis_thresh=1.65,
        unc_dis_thresh=2.5,
        ):
   
        map_col = torch.zeros(plan_cls.shape, dtype=torch.bool, device=plan_cls.device) 
        B, num_modes, fut_ts, _ = plan_preds.shape
        #choose boundry
        map_label_mask = torch.ne(map_det_label, 2)
        map_det_pts[map_label_mask] = 1e6
        map_det_betas[map_label_mask] = 1e6
        map_det_scores[map_label_mask] = 1e6
        #scores filter
        filter_map_mask = map_det_scores < map_score_thresh
        map_det_pts[filter_map_mask] = 1e6
        map_det_betas[filter_map_mask] = 1e6
        
        map_det_pts_flat = map_det_pts.view(-1, 2)
        map_det_betas_flat = map_det_betas.view(-1, 2)

        point_O_expanded = plan_preds.unsqueeze(3).expand(B, num_modes, fut_ts, map_det_pts_flat.size(0), 2)  # [B, 6, 6, 1000, 2]
        map_points_expanded = map_det_pts_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(point_O_expanded)  # [B, 6, 6, 1000, 2]
        map_beta_expanded = map_det_betas_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(point_O_expanded)  # [B, 6, 6, 1000, 2]

        a_vals_expanded = map_beta_expanded[..., 0]
        b_vals_expanded = map_beta_expanded[..., 1]

        inside_ellipses = ((point_O_expanded[..., 0] - map_points_expanded[..., 0])**2 / a_vals_expanded**2 +
                        (point_O_expanded[..., 1] - map_points_expanded[..., 1])**2 / b_vals_expanded**2) <= 1  # [B, 6, 6, 1000]

        if inside_ellipses.any():
            inside_idx = inside_ellipses.nonzero(as_tuple=True)
            selected_points = point_O_expanded[inside_idx]
            selected_centers = map_points_expanded[inside_idx]
            selected_a_vals = map_beta_expanded[inside_idx][..., 0]
            selected_b_vals = map_beta_expanded[inside_idx][..., 1]

            laplace_distribution = Laplace(selected_centers, torch.stack([selected_a_vals, selected_b_vals], dim=-1))
            nll = -laplace_distribution.log_prob(selected_points)
            nll_total = nll.sum(dim=-1) 
            nll_total_structured = torch.full_like(inside_ellipses, 1e6, dtype=nll.dtype)
            nll_total_structured[inside_idx] = nll_total 
            nll_total_flattened = nll_total_structured.view(B, num_modes, -1) 
            min_nll_per_mode = nll_total_flattened.min(dim=-1)[0] 

            map_col = min_nll_per_mode < unc_dis_thresh  # [B, num_modes]
        else:
            min_nll_per_mode = torch.full_like(plan_cls, 1e6, dtype=torch.float)
        
        num_map_col = map_col.int()

        map_col_wo_unc = self.detect_collision(plan_preds, map_det_pts, dis_thresh=dis_thresh)
        num_map_col_wo_unc = map_col_wo_unc.int().sum(dim=-1)
        map_col_wo_unc = map_col_wo_unc.any(dim=-1) 

        if map_col_wo_unc.all():
            map_col_wo_unc.fill_(False)

        final_col = torch.ones(plan_cls.shape, dtype=torch.bool, device=plan_cls.device) # [1, 6]

        num_col = num_map_col_wo_unc + num_map_col

        min_col_num_idx = torch.argmin(num_col, dim=-1)

        min_col_num = num_col[:, min_col_num_idx]

        mask = torch.eq(num_col, min_col_num)
        final_col[mask] = 0

        score_offset = final_col.float() * -999
        plan_cls = plan_cls + score_offset
        
        return plan_cls,min_nll_per_mode

    def rescore(
        self, 
        plan_cls,
        plan_preds, 
        map_det_pts,
        map_det_betas,
        map_det_label,
        map_det_scores,
        motion_cls,
        motion_pred, 
        det_bbox,
        det_confidence,
        score_thresh=0.4,
        static_dis_thresh=0.6,
        dim_scale=1.05,
        num_motion_mode=1,
        offset=0.4,
        map_score_thresh=0.4,
        dis_thresh=1.65,
        unc_dis_thresh=2.5
    ):

        map_col = torch.zeros(plan_cls.shape, dtype=torch.bool, device=plan_cls.device)
        B, num_modes, fut_ts, _ = plan_preds.shape
        #choose boundry
        map_label_mask = torch.ne(map_det_label, 2)
        map_det_pts[map_label_mask] = 1e6
        map_det_betas[map_label_mask] = 1e6
        map_det_scores[map_label_mask] = 1e6
        #scores filter
        filter_map_mask = map_det_scores < map_score_thresh
        map_det_pts[filter_map_mask] = 1e6
        map_det_betas[filter_map_mask] = 1e6

        map_det_pts_flat = map_det_pts.view(-1, 2)
        map_det_betas_flat = map_det_betas.view(-1, 2)

        point_O_expanded = plan_preds.unsqueeze(3).expand(B, num_modes, fut_ts, map_det_pts_flat.size(0), 2)
        map_points_expanded = map_det_pts_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(point_O_expanded)
        map_beta_expanded = map_det_betas_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(point_O_expanded)

        a_vals_expanded = map_beta_expanded[..., 0]
        b_vals_expanded = map_beta_expanded[..., 1]


        inside_ellipses = ((point_O_expanded[..., 0] - map_points_expanded[..., 0])**2 / a_vals_expanded**2 +
                        (point_O_expanded[..., 1] - map_points_expanded[..., 1])**2 / b_vals_expanded**2) <= 1 

        if inside_ellipses.any():
            inside_idx = inside_ellipses.nonzero(as_tuple=True) 

            points_inside_list = []
            for mode in inside_idx[1].unique():
                mode_idx = inside_idx[1] == mode
                points_inside_list.append(mode_idx.sum().item())

            selected_points = point_O_expanded[inside_idx]
            selected_centers = map_points_expanded[inside_idx]
            selected_a_vals = map_beta_expanded[inside_idx][..., 0]
            selected_b_vals = map_beta_expanded[inside_idx][..., 1]

            laplace_distribution = Laplace(selected_centers, torch.stack([selected_a_vals, selected_b_vals], dim=-1))
            nll = -laplace_distribution.log_prob(selected_points)
            nll_total = nll.sum(dim=-1) 
            nll_total_structured = torch.full_like(inside_ellipses, 1e6, dtype=nll.dtype) 
            nll_total_structured[inside_idx] = nll_total  
            nll_total_flattened = nll_total_structured.view(B, num_modes, -1)  
            min_nll_per_mode = nll_total_flattened.min(dim=-1)[0]

            map_col = min_nll_per_mode < unc_dis_thresh  # [B, num_modes]
 
            if map_col.all():
 
                map_col.fill_(False)
        else:
            min_nll_per_mode = torch.full_like(map_col, 1e6, dtype=torch.float) 
        num_map_col = map_col.int()

        map_col_wo_unc = self.detect_collision(plan_preds, map_det_pts, dis_thresh=dis_thresh) #[1,6,6]
        num_map_col_wo_unc = map_col_wo_unc.int().sum(dim=-1)
        map_col_wo_unc = map_col_wo_unc.any(dim=-1) #[1,6]


        if map_col_wo_unc.all():
            map_col_wo_unc.fill_(False)

        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat
        
        def get_yaw(traj, start_yaw=np.pi/2):
            yaw = traj.new_zeros(traj.shape[:-1])   #[1,6,7]
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)
        
        ## ego
        bs = 1  #only support batch_size=1 for testing

        plan_preds_cat = cat_with_zero(plan_preds)
        ego_box = det_bbox.new_zeros(bs, self.fut_mode, self.fut_ts + 1, 7)
        ego_box[..., :2] = plan_preds_cat 
        ego_box[..., 3:6] = ego_box.new_tensor([4.084, 1.85, 1.56]) * dim_scale  
        ego_box[..., 6:7] = get_yaw(plan_preds_cat) 

        ## motion
        motion_pred = motion_pred.view(bs, -1, self.fut_mode, self.fut_ts, 2) 
        motion_pred = motion_pred[..., :self.fut_ts, :].cumsum(-2)
        motion_pred = cat_with_zero(motion_pred) + det_bbox[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.fut_ts + 1, 2)
        motion_pred = torch.gather(motion_pred, 2, motion_mode_idx) 

        motion_box = motion_pred.new_zeros(motion_pred.shape[:-1] + (7,))
        motion_box[..., :2] = motion_pred    

        motion_box[..., 3:6] = det_bbox[..., None, None, 3:6]
        box_yaw = det_bbox[..., 6:7]  
        motion_box[..., 6:7] = get_yaw(motion_pred, box_yaw)  

        filter_mask = det_confidence < score_thresh 
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        ego_box[0] += offset * torch.cos(ego_box[6])
        ego_box[1] += offset * torch.sin(ego_box[6])
        col = self.check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)

        col = col.flatten(2, -1) 
        num_col_with_agents = col.int().sum(dim=-1) 
        col = col.any(dim=-1) 
        all_col = col.all(dim=-1) 
        col[all_col] = False 

        final_col = torch.ones(plan_cls.shape, dtype=torch.bool, device=plan_cls.device) 
        num_col = num_col_with_agents + num_map_col + num_map_col_wo_unc
        min_col_num_idx = torch.argmin(num_col, dim=-1)
        min_col_num = num_col[:, min_col_num_idx] 
        mask = torch.eq(num_col, min_col_num)
        final_col[mask] = 0

        if final_col.any():
            col = col.int()
            map_col_wo_unc = map_col_wo_unc.int()
            map_col = map_col.int()

        score_offset = final_col.float() * -999
        plan_cls = plan_cls + score_offset
        
        return plan_cls,min_nll_per_mode

    def detect_collision(self, plan_preds, target, dis_thresh=0.5):
        """
        Detect whether the trajectory collides with the map boundary.

        Args:
            plan_preds (torch.Tensor): ego_fut_preds, [1,6,6,2].
            target (torch.Tensor): lane_bound_preds, [num_vec, num_pts, 2].
            dis_thresh (float): Distance threshold for collision.

        Returns:
            torch.Tensor: Collision tensor [B, fut_ts], 1 indicates collision, 0 no collision.
        """
        B, num_modes, fut_ts, _ = plan_preds.shape
        ego_traj_starts = plan_preds[:, :, :-1, :]  # trajectory starting points
        ego_traj_ends = plan_preds  # trajectory ending points
        
        padding_zeros = torch.zeros((B, num_modes, 1, 2), dtype=plan_preds.dtype, device=plan_preds.device)  # initial position
        ego_traj_starts = torch.cat((padding_zeros, ego_traj_starts), dim=-2) 

        V, P, _ = target.size() 
        ego_traj_expanded = ego_traj_ends.unsqueeze(3).unsqueeze(4) 
        maps_expanded = target.unsqueeze(0).unsqueeze(0).unsqueeze(0)  

        # Calculate the Euclidean distance between trajectory points and map boundary points
        dist = torch.linalg.norm(ego_traj_expanded - maps_expanded, dim=-1)  
        min_dist = dist.min(dim=-1, keepdim=False)[0] 
        min_inst_idxs = torch.argmin(min_dist, dim=-1, keepdim=False).tolist()   
        batch_idxs = [[i] for i in range(dist.shape[0])]
        mode_idxs = [[i] for i in range(dist.shape[1])]
        ts_idxs = [[i for i in range(dist.shape[2])] for j in range(dist.shape[0])]

        bd_target = target.unsqueeze(0).unsqueeze(0).unsqueeze(1).repeat(1, num_modes, fut_ts, 1, 1, 1) 
        min_bd_insts = bd_target[batch_idxs, mode_idxs, ts_idxs, min_inst_idxs] 

        # Get boundary segments (start and end points)
        bd_inst_starts = min_bd_insts[:, :, :, :-1, :].flatten(0, 3)  
        bd_inst_ends = min_bd_insts[:, :, :, 1:, :].flatten(0, 3)   
        
        # Flatten trajectory segments
        ego_traj_starts_flat = ego_traj_starts.unsqueeze(-2).repeat(1, 1, 1, P-1, 1).flatten(0, 3)
        ego_traj_ends_flat = ego_traj_ends.unsqueeze(-2).repeat(1, 1, 1, P-1, 1).flatten(0, 3)
        
     
        intersect_mask = self.segments_intersect(ego_traj_starts_flat, ego_traj_ends_flat,
                                            bd_inst_starts, bd_inst_ends)
        intersect_mask = intersect_mask.reshape(B, num_modes, fut_ts, P-1)  
        intersect_mask = intersect_mask.any(dim=-1)  
        
     
        target = target.view(B, 1, -1, target.shape[-1])   
      
        dist = torch.linalg.norm(plan_preds[:, :, :, None, :] - target[:, :, None, :, :], dim=-1)
        min_idxs = torch.argmin(dist, dim=-1).tolist()  
        batch_idxs = [[i] for i in range(dist.shape[0])]
        mode_idxs = [[i] for i in range(dist.shape[1])]
        ts_idxs = [[i for i in range(dist.shape[2])] for j in range(dist.shape[0])]
        min_dist = dist[batch_idxs, mode_idxs, ts_idxs, min_idxs]
        dist_collision = min_dist <= dis_thresh  

        collision = torch.logical_or(dist_collision, intersect_mask) 

        collision = collision.int()    
        
        return collision


    def segments_intersect(self, line1_start, line1_end, line2_start, line2_end):
        """
        Detect if two line segments intersect.

        Args:
            line1_start, line1_end: Start and end points of the first line segment.
            line2_start, line2_end: Start and end points of the second line segment.

        Returns:
            torch.Tensor: Intersection mask [B*T*(P-1)].
        """
        # Calculate vector differences
        dx1 = line1_end[:, 0] - line1_start[:, 0]
        dy1 = line1_end[:, 1] - line1_start[:, 1]
        dx2 = line2_end[:, 0] - line2_start[:, 0]
        dy2 = line2_end[:, 1] - line2_start[:, 1]

        # Calculate determinant to check if lines are parallel
        det = dx1 * dy2 - dx2 * dy1
        det_mask = det != 0  # lines are not parallel

        # Calculate the intersection parameters (t1, t2)
        t1 = ((line2_start[:, 0] - line1_start[:, 0]) * dy2 
            - (line2_start[:, 1] - line1_start[:, 1]) * dx2) / det
        t2 = ((line2_start[:, 0] - line1_start[:, 0]) * dy1 
            - (line2_start[:, 1] - line1_start[:, 1]) * dx1) / det

        # Check if the intersection happens within the segments (0 <= t <= 1)
        intersect_mask = torch.logical_and(
            torch.logical_and(t1 >= 0, t1 <= 1),
            torch.logical_and(t2 >= 0, t2 <= 1)
        )

        # Handle parallel lines (no intersection)
        intersect_mask[det_mask == False] = False

        return intersect_mask


    def check_collision(self, boxes1, boxes2):
        '''
            A rough check for collision detection: 
                check if any corner point of boxes1 is inside boxes2 and vice versa.
            
            boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
            boxes2: tensor with shape [N, 7]
        '''
        col_1 = self.corners_in_box(boxes1.clone(), boxes2.clone())
        col_2 = self.corners_in_box(boxes2.clone(), boxes1.clone())
        collision = torch.logical_or(col_1, col_2)

        return collision

    def corners_in_box(self, boxes1, boxes2):
        if  boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
            return False

        boxes1_yaw = boxes1[:, 6].clone()
        boxes1_loc = boxes1[:, :3].clone()
        cos_yaw = torch.cos(-boxes1_yaw)
        sin_yaw = torch.sin(-boxes1_yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )
        # translate and rotate boxes
        boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
        boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
        boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

        boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
        boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
        boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

        corners_box2 = self.box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
        corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
        L = boxes1[:, [3]]
        W = boxes1[:, [4]]

        collision = torch.logical_and(
            torch.logical_and(corners_box2[..., 0] <= L / 2, corners_box2[..., 0] >= -L / 2),
            torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
        )
        collision = collision.any(dim=-1)

        return collision

    def box3d_to_corners(self, box3d):
        if isinstance(box3d, torch.Tensor):
            box3d = box3d.detach().cpu().numpy()
        corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
        corners = box3d[:, None, 3:6] * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        rot_cos = np.cos(box3d[:, 6])
        rot_sin = np.sin(box3d[:, 6])
        rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
        rot_mat[:, 0, 0] = rot_cos
        rot_mat[:, 0, 1] = -rot_sin
        rot_mat[:, 1, 0] = rot_sin
        rot_mat[:, 1, 1] = rot_cos
        corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
        corners += box3d[:, None, :3]
        return corners

    def map_pred2result(self, bboxes, scores, labels, pts, betas, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'),
            map_betas_3d=betas.to('cpu'))    

        if attrs is not None:
            result_dict['map_attrs_3d'] = attrs.cpu()

        return result_dict

    def assign_pred_to_gt_vip3d(
        self,
        bbox_result,
        gt_bbox,
        gt_label,
        match_dis_thresh=2.0
    ):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """     
        dynamic_list = [0,1,3,4,6,7,8]
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result['boxes_3d'].center[:, :2]
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]

        return matched_bbox_result

    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        
        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met+'_'+cls] = 0.0

        veh_list = [0,1,3,4]
        ignore_list = ['construction_vehicle', 'barrier',
                       'traffic_cone', 'motorcycle', 'bicycle']

        for i in range(pred_bbox['labels_3d'].shape[0]):  
            pred_bbox['labels_3d'][i] = 0 if pred_bbox['labels_3d'][i] in veh_list else pred_bbox['labels_3d'][i] 
            box_name = mapped_class_names[pred_bbox['labels_3d'][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict['fp_'+box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts*2:self.fut_ts*3]
            num_valid_ts = sum(gt_fut_masks==1)
            if num_valid_ts == self.fut_ts:
                metric_dict['gt_'+box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict['cnt_ade_'+box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][:self.fut_ts*2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict['ADE_'+box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict['cnt_fde_'+box_name] += 1
                    metric_dict['FDE_'+box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict['hit_'+box_name] += 1
                    else:
                        metric_dict['MR_'+box_name] += 1

        return metric_dict

    ### same planning metric as stp3
    def compute_planner_metric_stp3(
        self,
        sample_idx,
        pred_ego_fut_trajs,
        pred_ego_trajs_cls_scores,
        all_pred_ego_fut_trajs,
        all_pred_ego_trajs_cls_scores,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                final_drivable_area_conflict = self.planning_metric.calculate_drivable_area_conflict(
                    self.nuscene_map,
                    self.sampleidx2index[sample_idx],
                    self.pts,
                    pred_ego_fut_trajs[:,:cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time]
                )
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
                metric_dict['final_plan_DAC_{}s'.format(i+1)]= final_drivable_area_conflict.mean().item()
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
                metric_dict['final_plan_DAC_{}s'.format(i+1)]=0.0
            
        return metric_dict

    def set_epoch(self, epoch): 
        self.pts_bbox_head.epoch = epoch