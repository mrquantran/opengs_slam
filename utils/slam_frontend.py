import time
import math

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.dust3r_utils import get_result, get_scale

def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B,P,1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask

def depths_to_points(view, depthmap):
    # c2w = (view.world_view_transform.T).inverse()
    # we train in camera coordinate
    c2w = torch.eye(4).float().cuda()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().cuda()
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

class FrontEnd(mp.Process):
    def __init__(self, config, d3r_model):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        self.d3r_model = d3r_model
        self.last_color = None          # last frame RGB
        self.pts3d = None               # last frame pointcloud
        self.imgs = None
        self.mask = None
        self.matches_im0 = None
        self.matches_im1 = None
        self.matches_3d0 = None
        self.scale = 1                  # Scale factor computed using median, not enabled
        self.scale1 = 1                 # Scale factor computed using mean, used for scale correction
        self.theta = 0                  # Camera angle diff from last keyframe

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        if len(self.kf_indices) > 0:
            last_kf = self.kf_indices[-1]
            viewpoint_last = self.cameras[last_kf]
            R_last = viewpoint_last.R
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        R_now = viewpoint.R
        # Compute angle diff from previous frame
        if len(self.kf_indices) > 1:
            R_now = R_now.to(torch.float32)
            R_last = R_last.to(torch.float32)
            R_diff = torch.matmul(R_last.T, R_now)
            trace_R_diff = torch.trace(R_diff)
            theta_rad = torch.acos((trace_R_diff - 1) / 2)
            theta_deg = torch.rad2deg(theta_rad)
            self.theta = theta_deg
        # print("angle diff is:",self.theta)
        ### MonoGS Gaussian init depth, not used
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0
            return initial_depth.cpu().numpy()[0]

        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()      # (C, H, W), not used!

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def find_nearest_keyframe(self, current_viewpoint, window_indices):
        """
        Finds the keyframe in the current window that is closest in position
        to the current viewpoint.
        """
        # 1. Kiểm tra điều kiện biên
        if len(window_indices) <= 1:
            return None

        # 2. Lấy vị trí camera hiện tại
        current_pos = current_viewpoint.camera_center

        best_kf_idx = -1
        min_dist = float('inf')

        # 3. Duyệt qua các keyframe trong cửa sổ
        # We iterate starting from the second keyframe in the window,
        # as the first one is the newest one (which might be the current frame itself
        # if it becomes a keyframe, or the one right before it).
        for kf_idx in window_indices[1:]:
            keyframe_cam = self.cameras.get(kf_idx)
            if keyframe_cam:
                kf_pos = keyframe_cam.camera_center
                dist = torch.norm(current_pos - kf_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_kf_idx = kf_idx

        # 4. Trả về kết quả
        if best_kf_idx != -1:
            return self.cameras.get(best_kf_idx)
        else:
            return None

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        # Get relative pose, pointcloud, and point matching correspondence
        trans_pose ,pts3d, imgs, matches_im0, matches_im1, matches_3d0=get_result(viewpoint.original_image, self.last_color, model=self.d3r_model, device=self.device)
        self.pts3d = pts3d
        self.imgs = imgs
        # Get scale factor
        scale1, scale = get_scale(self.matches_im1, self.matches_im0, matches_im1, matches_im0, self.matches_3d0, matches_3d0)
        self.scale = self.scale * scale
        self.scale1 = self.scale1 * scale1
        self.matches_im0 = matches_im0
        self.matches_im1 = matches_im1
        self.matches_3d0 = matches_3d0
        # Pose Estimation
        trans_pose_inv = np.linalg.inv(trans_pose)
        trans_pose_inv_torch = torch.from_numpy(trans_pose_inv).to(self.device)
        w2c1 = getWorld2View2(prev.R, prev.T)
        w2c2 = trans_pose_inv_torch @ w2c1
        viewpoint.update_RT(w2c2[:3,:3],w2c2[:3,3])         # Compute current frame pose estimation using relative pose
        # pose optimization
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)

        nearest_cam = self.find_nearest_keyframe(viewpoint, self.current_window)

        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )

            ncc_loss = torch.tensor(0.0, device=self.device)
            if nearest_cam is not None:
                # 1. Lấy thông tin cần thiết từ render_pkg
                # Giả sử rasterizer trả về 'normal'. Nếu không, ta cần tính nó.
                # rendered_normal = render_pkg.get("normal")
                # if rendered_normal is None:

                rendered_normal, _ = depth_to_normal(viewpoint, depth.unsqueeze(0))
                rendered_normal = rendered_normal.squeeze(0).permute(2, 0, 1) # Chuyển về (C, H, W)

                # 2. Lấy mẫu pixel và tính toán consistency
                patch_size = 3 # Lấy từ config hoặc hardcode
                sample_num = 102400 # Lấy từ config hoặc hardcode
                H, W = depth.shape[-2:]

                # Ưu tiên lấy mẫu ở vùng có gradient cao để NCC hiệu quả hơn
                valid_pixels_mask = (depth > 0).squeeze()
                high_grad_mask = viewpoint.grad_mask.squeeze() > 0
                final_sampling_mask = valid_pixels_mask & high_grad_mask

                valid_indices = torch.where(final_sampling_mask.view(-1))[0]

                if valid_indices.numel() > sample_num:
                    sampled_indices = valid_indices[torch.randperm(valid_indices.numel(), device=self.device)[:sample_num]]
                else:
                    sampled_indices = valid_indices

                if sampled_indices.numel() > 0:
                    # --- B. Tính Homography (Differentiable) ---
                    # Chuyển chỉ số 1D thành tọa độ 2D
                    sampled_pixels_y = sampled_indices // W
                    sampled_pixels_x = sampled_indices % W
                    sampled_pixels = torch.stack((sampled_pixels_x, sampled_pixels_y), dim=-1).float()

                    # Lấy normal tại các điểm đã lấy mẫu
                    ref_local_n = rendered_normal.permute(1, 2, 0).reshape(-1, 3)[sampled_indices]

                    # Tính Plane Distance `d`
                    rays_d = viewpoint.get_rays()
                    rendered_normal_flat = rendered_normal.permute(1, 2, 0).reshape(-1, 3)
                    plane_distance = depth.reshape(-1) * torch.abs((rendered_normal_flat * rays_d.reshape(-1, 3)).sum(dim=-1))
                    ref_local_d = plane_distance[sampled_indices]

                    # Ma trận extrinsic tương đối
                    ref_to_nearest_w2c = nearest_cam.world_view_transform.T @ torch.linalg.inv(viewpoint.world_view_transform.T)
                    ref_to_nearest_r = ref_to_nearest_w2c[:3, :3]
                    ref_to_nearest_t = ref_to_nearest_w2c[:3, 3]

                    # Công thức Homography H = K_n * (R - t*n^T/d) * K_r^{-1}
                    H_ref_to_neareast = ref_to_nearest_r[None] - \
                        (ref_to_nearest_t[None, :, None] @ ref_local_n[:, None, :]) / (ref_local_d[:, None, None] + 1e-8)

                    # Áp dụng ma trận nội suy
                    K_ref_inv = torch.linalg.inv(viewpoint.get_k())
                    K_nea = nearest_cam.get_k()
                    H_ref_to_neareast = (K_nea @ H_ref_to_neareast) @ K_ref_inv

                    # --- C. Warp Patches và Tính NCC Loss ---
                    h_patch_size = (patch_size - 1) // 2
                    offsets = patch_offsets(h_patch_size, self.device)
                    ref_pixels_patch = sampled_pixels.reshape(-1, 1, 2) + offsets.float()

                    # Lấy patch từ ảnh GT của view hiện tại
                    _, ref_image_gray = viewpoint.get_image()

                    ref_pixels_patch_norm = ref_pixels_patch.clone()
                    ref_pixels_patch_norm[..., 0] = (ref_pixels_patch_norm[..., 0] / (W - 1)) * 2 - 1
                    ref_pixels_patch_norm[..., 1] = (ref_pixels_patch_norm[..., 1] / (H - 1)) * 2 - 1
                    ref_gray_val = F.grid_sample(
                        ref_image_gray.unsqueeze(0),
                        ref_pixels_patch_norm.unsqueeze(0),
                        align_corners=True
                    ).squeeze(0).squeeze(0).view(-1, patch_size**2)

                    # Warp tọa độ và lấy patch từ ảnh GT của view lân cận
                    nea_pixels_patch = patch_warp(H_ref_to_neareast, ref_pixels_patch)
                    _, nea_image_gray = nearest_cam.get_image()

                    nea_pixels_patch_norm = nea_pixels_patch.clone()
                    nea_pixels_patch_norm[..., 0] = (nea_pixels_patch_norm[..., 0] / (W - 1)) * 2 - 1
                    nea_pixels_patch_norm[..., 1] = (nea_pixels_patch_norm[..., 1] / (H - 1)) * 2 - 1
                    nea_gray_val = F.grid_sample(
                        nea_image_gray.unsqueeze(0),
                        nea_pixels_patch_norm.unsqueeze(0),
                        align_corners=True
                    ).squeeze(0).squeeze(0).view(-1, patch_size**2)

                    # Tính NCC loss
                    ncc_val, _ = lncc(ref_gray_val, nea_gray_val)
                    ncc_loss = ncc_val.mean()

            # 4. TỔNG HỢP LOSS VÀ BACKPROPAGATION
            lambda_photo_mv = 0.15 # Nên lấy từ config
            print(f"\rNCC Loss: {ncc_loss.item():.6f}", end="", flush=True)

            total_loss = loss_tracking + lambda_photo_mv * ncc_loss
            total_loss.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        ## Print camera pose change and scale factor
        #c2w1 = torch.linalg.inv(w2c1)
        #c2w2 = torch.linalg.inv(w2c2)
        #c2w3 = torch.linalg.inv(getWorld2View2(viewpoint.R, viewpoint.T))
        #print("Pose estimation translation:", c2w2[:3,3]- c2w1[:3,3])
        #print("Optimized pose translation:", c2w3[:3,3]- c2w2[:3,3])
        #print("Current frame mean scale:",scale1)
        #print("Cumulative mean scale factor:", self.scale1)
        #print("Current frame median scale:",scale)
        #print("Cumulative median scale factor:", self.scale)

        self.median_depth = get_median_depth(depth, opacity)    # Median rendered depth for keyframe determination
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)
        # Remove earliest keyframe with overlap below threshold
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))
        # If the window count exceeds the limit, remove the frame farthest from the current frame.
        # The distance is weighted to favor deleting the candidate with the highest inverse distance to the other candidates.
        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)
        #print("current keyframe ",cur_frame_idx,'window is ',window)
        return window, removed_frame
    ### Exchange info with backend via following functions
    # Request new keyframe; enqueue related info to backend
    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, self.pts3d, self.imgs, self.mask, self.scale1, self.theta]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1
    # Request initialization; enqueue related info to backend.
    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map, self.pts3d, self.imgs, self.mask, self.scale1]
        self.backend_queue.put(msg)
        self.requested_init = True
    # Sync data from backend (3D Gaussians, occlusion-aware visibility, keyframe info)
    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    # Main loop: process messages in frontend and backend queues; perform tracking, keyframe management;
    # synchronize data, clean up resources, and save results
    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():         # Check if frontend_queue is empty; if so, start processing the current frame
                tic.record()
                if cur_frame_idx >= len(self.dataset):  # Finish the frontend process
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.last_color = self.cameras[cur_frame_idx].original_image
                    _ ,pts3d, imgs, self.matches_im0, self.matches_im1, self.matches_3d0=get_result(self.last_color,self.last_color, model=self.d3r_model, device=self.device)
                    self.pts3d = pts3d
                    self.imgs = imgs
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                self.last_color = self.cameras[cur_frame_idx].original_image

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    Log("new keyframe: ", cur_frame_idx)
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (                    # Perform trajectory evaluation when certain conditions are met
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:       # If the frontend queue contains messages from the backend, process them
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
