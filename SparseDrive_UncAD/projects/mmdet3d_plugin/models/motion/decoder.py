from typing import Optional

import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.detection3d.decoder import *
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from torch.distributions.laplace import Laplace


@BBOX_CODERS.register_module()
class SparseBox3DMotionDecoder(SparseBox3DDecoder):
    def __init__(self):
        super(SparseBox3DMotionDecoder, self).__init__()

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        motion_output=None,
        output_idx=-1,
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality[output_idx] is None:
            quality = None
        if quality is not None:
            centerness = quality[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        anchor_queue = motion_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = motion_output["period"]

        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = decode_box(box)
            trajs = motion_output["prediction"][-1]
            traj_cls = motion_output["classification"][-1].sigmoid()
            traj = trajs[i, indices[i] // num_cls]
            traj_cls = traj_cls[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                traj = traj[mask[i]]
                traj_cls = traj_cls[mask[i]]
            traj = traj.cumsum(dim=-2) + box[:, None, None, :2]
            output.append(
                {
                    "trajs_3d": traj.cpu(),
                    "trajs_score": traj_cls.cpu()
                }
            )

            temp_anchor = anchor_queue[i, indices[i] // num_cls]
            temp_period = period[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                temp_anchor = temp_anchor[mask[i]]
                temp_period = temp_period[mask[i]]
            num_pred, queue_len = temp_anchor.shape[:2]
            temp_anchor = temp_anchor.flatten(0, 1)
            temp_anchor = decode_box(temp_anchor)
            temp_anchor = temp_anchor.reshape([num_pred, queue_len, box.shape[-1]])
            output[-1]['anchor_queue'] = temp_anchor.cpu()
            output[-1]['period'] = temp_period.cpu()
        
        return output


@BBOX_CODERS.register_module()
class HierarchicalPlanningDecoder(object):
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        use_rescore=False,
    ):
        super(HierarchicalPlanningDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.use_rescore = use_rescore
    
    def decode(
        self, 
        map_output,
        det_output,
        motion_output,
        planning_output, 
        data,
    ):
        classification = planning_output['classification'][-1]
        prediction = planning_output['prediction'][-1]
        bs = classification.shape[0]
        classification = classification.reshape(bs, 3, self.ego_fut_mode)
        prediction = prediction.reshape(bs, 3, self.ego_fut_mode, self.ego_fut_ts, 2).cumsum(dim=-2)
        classification, final_planning = self.select(map_output,det_output, motion_output, classification, prediction, data)
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        output = []
        for i, (cls, pred) in enumerate(zip(classification, prediction)):
            output.append(
                {
                    "planning_score": cls.sigmoid().cpu(),
                    "planning": pred.cpu(),
                    "final_planning": final_planning[i].cpu(),
                    "ego_period": period[i].cpu(),
                    "ego_anchor_queue": decode_box(anchor_queue[i]).cpu(),
                }
            )

        return output

    def select(
        self,
        map_output,
        det_output,
        motion_output,
        plan_cls,
        plan_reg,
        data,
    ):

        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]
        
        # cmd select
        bs = motion_cls.shape[0]
        bs_indices = torch.arange(bs, device=motion_cls.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, cmd]
        plan_reg = plan_reg[bs_indices, cmd]

        # rescore
        if self.use_rescore:
            plan_cls = self.rescore(
                map_output,
                plan_cls,
                plan_reg, 
                motion_cls,
                motion_reg, 
                det_anchors,
                det_confidence,
            )
        plan_cls_full[bs_indices, cmd] = plan_cls
        mode_idx = plan_cls.argmax(dim=-1)
        final_planning = plan_reg[bs_indices, mode_idx]
        return plan_cls_full, final_planning

    def rescore(
        self, 
        map_output,
        plan_cls,
        plan_reg, 
        motion_cls,
        motion_reg, 
        det_anchors,
        det_confidence,
        score_thresh=0.35,
        map_score_thresh=0.5,
        static_dis_thresh=0.55,
        dim_scale=1.2,
        num_motion_mode=1,
        offset=0.4,
        dis_thresh=1.4,
    ):
        
        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat
        
        def get_yaw(traj, start_yaw=np.pi/2):
            yaw = traj.new_zeros(traj.shape[:-1])
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
        #1.map var:
        map_det_pts=map_output['prediction'][-1].squeeze(0).view(100,20,2)
        map_det_betas=map_output['prediction_beta'][-1].squeeze(0).view(100,20,2)
        cls_scores=map_output['classification'][-1].squeeze(0)
        map_det_scores, map_det_label=cls_scores.topk(k=1,dim=1)
        map_det_scores = map_det_scores.squeeze(1)
        map_det_label = map_det_label.squeeze(1)
        #2.map col select:
        #===============================add map col=============================#
        map_col = torch.zeros(plan_cls.shape, dtype=torch.bool, device=plan_cls.device) # [1, 6]
        B, num_modes, fut_ts, _ = plan_reg.shape
        #choose boundry
        map_label_mask = torch.ne(map_det_label, 2)
        map_det_pts[map_label_mask] = 1e6
        map_det_betas[map_label_mask] = 1e6
        map_det_scores[map_label_mask] = 1e6
        #scores filter
        filter_map_mask = map_det_scores < map_score_thresh
        map_det_pts[filter_map_mask] = 1e6
        map_det_betas[filter_map_mask] = 1e6

        map_det_pts_flat = map_det_pts.reshape(-1, 2) 
        map_det_betas_flat = map_det_betas.reshape(-1, 2) 

        point_O_expanded = plan_reg.unsqueeze(3).expand(B, num_modes, fut_ts, map_det_pts_flat.size(0), 2) #[B, 6, 6, 1000, 2]
        map_points_expanded = map_det_pts_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(point_O_expanded) #[B, 6, 6, 1000, 2]
        map_beta_expanded = map_det_betas_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(point_O_expanded) #[B, 6, 6, 1000, 2]

        a_vals_expanded = map_beta_expanded[..., 0]
        b_vals_expanded = map_beta_expanded[..., 1]

        inside_ellipses = ((point_O_expanded[..., 0] - map_points_expanded[..., 0])**2 / a_vals_expanded**2 +
                        (point_O_expanded[..., 1] - map_points_expanded[..., 1])**2 / b_vals_expanded**2) <= 1 #[B, 6, 6, 1000]

        if inside_ellipses.any():#[B, 6, 6, 1000]
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
          
            unc_dis_thresh=2.5 

            map_col = min_nll_per_mode < unc_dis_thresh  # [B, num_modes]
        else:
            min_nll_per_mode = torch.full_like(map_col, 1e6, dtype=torch.float) 
        num_map_col = map_col.int()
        
        map_col_wo_unc = self.detect_collision(plan_reg, map_det_pts, dis_thresh=dis_thresh) #[1,6,6]
        num_map_col_wo_unc = map_col_wo_unc.int().sum(dim=-1)
        map_col_wo_unc = map_col_wo_unc.any(dim=-1) #[1,6]

        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

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
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        
        col = col.flatten(2, -1)
        num_col_with_agents = col.int().sum(dim=-1) 
        col = col.any(dim=-1) 
        all_col = col.all(dim=-1) 

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
        
        return plan_cls
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
        ego_traj_starts = plan_preds[:, :, :-1, :]  
        ego_traj_ends = plan_preds  
        

        padding_zeros = torch.zeros((B, num_modes, 1, 2), dtype=plan_preds.dtype, device=plan_preds.device)
        ego_traj_starts = torch.cat((padding_zeros, ego_traj_starts), dim=-2)  

        V, P, _ = target.size() 
        ego_traj_expanded = ego_traj_ends.unsqueeze(3).unsqueeze(4)  
        maps_expanded = target.unsqueeze(0).unsqueeze(0).unsqueeze(0) 

        dist = torch.linalg.norm(ego_traj_expanded - maps_expanded, dim=-1)  
        min_dist = dist.min(dim=-1, keepdim=False)[0]  
        min_inst_idxs = torch.argmin(min_dist, dim=-1, keepdim=False).tolist()   
        batch_idxs = [[i] for i in range(dist.shape[0])]
        mode_idxs = [[i] for i in range(dist.shape[1])]
        ts_idxs = [[i for i in range(dist.shape[2])] for j in range(dist.shape[0])]

        bd_target = target.unsqueeze(0).unsqueeze(0).unsqueeze(1).repeat(1, num_modes, fut_ts, 1, 1, 1)    #[[B, num_modes, fut_ts, V, P, 2]
        min_bd_insts = bd_target[batch_idxs, mode_idxs, ts_idxs, min_inst_idxs]  # [B, num_modes, fut_ts, P, 2]


        bd_inst_starts = min_bd_insts[:, :, :, :-1, :].flatten(0, 3)  
        bd_inst_ends = min_bd_insts[:, :, :, 1:, :].flatten(0, 3)       
        
        ego_traj_starts_flat = ego_traj_starts.unsqueeze(-2).repeat(1, 1, 1, P-1, 1).flatten(0, 3)
        ego_traj_ends_flat = ego_traj_ends.unsqueeze(-2).repeat(1, 1, 1, P-1, 1).flatten(0, 3)

        intersect_mask = self.segments_intersect(ego_traj_starts_flat, ego_traj_ends_flat,
                                            bd_inst_starts, bd_inst_ends)
        intersect_mask = intersect_mask.reshape(B, num_modes, fut_ts, P-1)  
        intersect_mask = intersect_mask.any(dim=-1) 

        target = target.view(B, 1, -1, target.shape[-1])    
        # [B, num_modes, fut_ts, num_vec*num_pts]
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

def check_collision(boxes1, boxes2):
    '''
        A rough check for collision detection: 
            check if any corner point of boxes1 is inside boxes2 and vice versa.
        
        boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
        boxes2: tensor with shape [N, 7]
    '''
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())
    collision = torch.logical_or(col_1, col_2)

    return collision

def corners_in_box(boxes1, boxes2):
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

    corners_box2 = box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
    corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    collision = torch.logical_and(
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
    )
    collision = collision.any(dim=-1)

    return collision