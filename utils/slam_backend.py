import random
import time

import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping


class BackEnd(mp.Process):
    def __init__(self, config, save_dir=None, pgo_queue_in=None, pgo_queue_out=None):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.pgo_queue_in = pgo_queue_in
        self.pgo_queue_out = pgo_queue_out
        self.live_mode = False
        self.save_dir = save_dir

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0    # Total iterations
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        self.pcd_scale = 1          # scale factor
        self.theta = 0              # Camera angle diff from last keyframe

        # PGO integration
        self.use_pgo = pgo_queue_in is not None and pgo_queue_out is not None
        self.keyframe_images = {}  # Store RGB images for PGO

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
    # In MonoGS, initialize Gaussians and add to the current scene (not enabled)
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )
    # Initialize Gaussians via pointmap and add to the current scene
    def add_next_kf_dust3r(self, frame_idx, pts3d, imgs, T, mask=None, init=False, scale=1):
        fused_point_cloud, features, scales, rots, opacities = (
            self.gaussians.create_pcd_from_dust3r(pts3d, imgs, T, frame_idx, self.save_dir, scale, mask, init=init)
        )
        self.gaussians.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, frame_idx
        )

    def reset(self):
        self.iteration_count = 0
        self.iteration_count1 = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    # Initialize SLAM map by optimizing Gaussian parameters over multiple iterations
    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            self.iteration_count1 += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, viewpoint, depth=depth,initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.iteration_count1 += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, viewpoint, depth=depth
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, viewpoint, depth=depth
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            # Compute isotropic loss and add it to the total loss
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Reset opacity every fixed number of iterations
                #if (self.iteration_count % self.gaussian_reset) == 0 and (
                #    not update_gaussian
                #):
                #    Log("Resetting the opacity of non-visible Gaussians")
                #    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                #    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(20000)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)
    # Main loop: process messages from the backend queue, perform map optimization, color refinement,
    # initialization, and keyframe management; synchronize data and push to the frontend
    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue
                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                # Check for PGO updates first
                self._check_pgo_updates()

                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    pts3d = data[4]
                    imgs = data[5]
                    mask = data[6]
                    self.scale = 1
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    #self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, init=True)
                    ## Initialize SLAM map using pointmap
                    self.add_next_kf_dust3r(cur_frame_idx, pts3d, imgs, T, mask, init=True, scale=self.scale)
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    pts3d = data[5]
                    imgs = data[6]
                    mask = data[7]
                    self.scale = data[8]
                    self.theta = data[9]

                    # Send keyframe to PGO thread if enabled
                    if self.use_pgo and len(data) > 10:
                        rgb_image = data[10]  # RGB image for landmark detection
                        if rgb_image is not None:
                            self.keyframe_images[cur_frame_idx] = rgb_image
                            self._send_keyframe_to_pgo(cur_frame_idx, viewpoint, rgb_image)
                        else:
                            Log(f"RGB image is None for keyframe {cur_frame_idx}, skipping PGO", tag="Backend")
                    theta_value = self.theta.item()
                    ## adjust the cumulative iterations for Adaptive Learning Rate Adjustment
                    if theta_value >= 2:
                        self.iteration_count = self.iteration_count * (1-np.sqrt(theta_value / 90))
                        self.iteration_count = int(self.iteration_count)
                        self.gaussians.update_learning_rate(self.iteration_count)
                    #print("current keyframe:", cur_frame_idx, "cumulative iterations:", self.iteration_count)

                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    #self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                    ## Adaptive Scale Mapper
                    self.add_next_kf_dust3r(cur_frame_idx, pts3d, imgs, T, mask, scale=self.scale)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 150
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return

    def _send_keyframe_to_pgo(self, keyframe_id, viewpoint, rgb_image):
        """Send keyframe data to PGO thread for loop closure detection"""
        if not self.use_pgo:
            return

        try:
            # Extract pose from viewpoint
            R = viewpoint.R.cpu().numpy()
            t = viewpoint.T.cpu().numpy()
            pose = [R, t]

            # Prepare keyframe data
            keyframe_data = {
                'keyframe_id': keyframe_id,
                'pose': pose,
                'rgb_image': rgb_image,
                'timestamp': time.time()
            }

            # Send to PGO thread
            self.pgo_queue_in.put(["new_keyframe", keyframe_data])
            Log(f"Sent keyframe {keyframe_id} to PGO thread", tag="Backend")

        except Exception as e:
            Log(f"Error sending keyframe to PGO: {e}", tag="Backend")

    def _check_pgo_updates(self):
        """Check for updates from PGO thread"""
        if not self.use_pgo:
            return

        try:
            while not self.pgo_queue_out.empty():
                data = self.pgo_queue_out.get()

                if data[0] == "optimized_poses":
                    optimized_poses = data[1]
                    self._apply_optimized_poses(optimized_poses)

        except Exception as e:
            Log(f"Error checking PGO updates: {e}", tag="Backend")

    def _apply_optimized_poses(self, optimized_poses):
        """Apply optimized poses from PGO to the system"""
        try:
            Log(f"Applying optimized poses for {len(optimized_poses)} keyframes", tag="Backend")

            # Update viewpoints with optimized poses
            for keyframe_id, optimized_pose in optimized_poses.items():
                if keyframe_id in self.viewpoints:
                    R_opt, t_opt = optimized_pose

                    # Convert to torch tensors
                    R_tensor = torch.from_numpy(R_opt).float().to(self.device)
                    t_tensor = torch.from_numpy(t_opt).float().to(self.device)

                    # Update viewpoint pose
                    viewpoint = self.viewpoints[keyframe_id]
                    viewpoint.R = R_tensor
                    viewpoint.T = t_tensor

                    # Update pose deltas (assuming they represent the difference from initial pose)
                    # This is a simplified approach - in practice you might need more sophisticated pose management
                    viewpoint.cam_rot_delta.data = torch.zeros_like(viewpoint.cam_rot_delta.data)
                    viewpoint.cam_trans_delta.data = torch.zeros_like(viewpoint.cam_trans_delta.data)

            # Update Gaussian map based on pose changes
            self._update_gaussian_map_poses(optimized_poses)

            Log("Successfully applied optimized poses", tag="Backend")

        except Exception as e:
            Log(f"Error applying optimized poses: {e}", tag="Backend")

    def _update_gaussian_map_poses(self, optimized_poses):
        """Update Gaussian map based on pose changes from PGO"""
        try:
            Log(f"Updating Gaussian map for {len(optimized_poses)} keyframes", tag="Backend")

            # For now, we'll just log the pose updates and let the Gaussian optimization
            # handle the rest. This avoids complex tensor operations that can cause gradient issues.

            for keyframe_id, optimized_pose in optimized_poses.items():
                if keyframe_id in self.viewpoints:
                    R_opt, t_opt = optimized_pose
                    viewpoint = self.viewpoints[keyframe_id]

                    # Log the pose change
                    t_change = np.linalg.norm(t_opt - viewpoint.T.cpu().numpy())
                    Log(f"Pose update for keyframe {keyframe_id}: translation change = {t_change:.4f}m", tag="Backend")

            # Reset Gaussian optimizer state to ensure it works with updated poses
            if hasattr(self, 'gaussians') and hasattr(self.gaussians, 'optimizer'):
                # Clear optimizer state to prevent conflicts with new poses
                for param_group in self.gaussians.optimizer.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None:
                            param.grad.data.zero_()

                Log("Reset Gaussian optimizer state after pose update", tag="Backend")

        except Exception as e:
            Log(f"Error updating Gaussian map poses: {e}", tag="Backend")
