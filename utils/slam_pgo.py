import torch
import torch.multiprocessing as mp
import numpy as np
import time
import cv2
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms

# Import YOLOv8 for object detection
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not available. Install with: pip install ultralytics")

# Import g2o for pose graph optimization
try:
    import g2o

    G2O_AVAILABLE = True
except ImportError:
    G2O_AVAILABLE = False
    print("Warning: g2o not available. Install with: pip install g2o-python")

from utils.logging_utils import Log
from utils.pose_utils import update_pose


class Landmark:
    """Class to represent a landmark detected in a keyframe"""

    def __init__(self, mask, class_id, class_name, bbox, descriptor, keyframe_id):
        self.mask = mask
        self.class_id = class_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.descriptor = descriptor  # 2048-dim feature vector
        self.keyframe_id = keyframe_id
        self.confidence = 0.0


class PoseGraphNode:
    """Class to represent a node in the pose graph"""

    def __init__(self, keyframe_id, pose, timestamp=None):
        self.keyframe_id = keyframe_id
        self.pose = pose  # [R, t] where R is 3x3 rotation matrix, t is 3x1 translation
        self.timestamp = timestamp
        self.landmarks = []  # List of Landmark objects


class PoseGraphEdge:
    """Class to represent an edge in the pose graph"""

    def __init__(
        self, from_id, to_id, relative_pose, edge_type, information_matrix=None
    ):
        self.from_id = from_id
        self.to_id = to_id
        self.relative_pose = relative_pose  # [R, t] relative pose
        self.edge_type = edge_type  # 'sequential' or 'loop_closure'
        self.information_matrix = (
            information_matrix if information_matrix is not None else np.eye(6)
        )


class PGOThread(mp.Process):
    """Pose Graph Optimization thread for loop closure detection and global optimization"""

    def __init__(self, pgo_queue_in, pgo_queue_out, config, save_dir=None):
        super().__init__()
        self.pgo_queue_in = pgo_queue_in
        self.pgo_queue_out = pgo_queue_out
        self.config = config
        self.save_dir = save_dir

        # Pose graph data structures
        self.pose_graph_nodes = {}  # keyframe_id -> PoseGraphNode
        self.pose_graph_edges = []  # List of PoseGraphEdge
        self.landmark_db = {}  # landmark_id -> Landmark

        # Models for landmark detection and feature extraction
        self.yolo_model = None
        self.feature_extractor = None
        self.feature_extractor_transform = None

        # Loop closure parameters
        self.loop_closure_threshold = 0.7  # Cosine similarity threshold
        self.min_loop_closure_distance = (
            10  # Minimum keyframe distance for loop closure
        )
        self.max_loop_closure_candidates = 5

        # Optimization parameters
        self.optimization_frequency = 5  # Optimize every N loop closures
        self.loop_closure_count = 0
        self.last_optimized_kf_id = -1  # Track last optimized keyframe

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize models
        self._initialize_models()

        Log("PGO Thread initialized", tag="PGO")

    def _initialize_models(self):
        """Initialize YOLOv8 and ResNet50 models"""
        # Initialize YOLOv8 for object detection
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO("yolov8l-seg.pt")
                Log("YOLOv8 model loaded successfully", tag="PGO")
            except Exception as e:
                Log(f"Failed to load YOLOv8 model: {e}", tag="PGO")
                self.yolo_model = None
        else:
            Log("YOLOv8 not available, landmark detection disabled", tag="PGO")

        # Initialize ResNet50 for feature extraction
        try:
            self.feature_extractor = models.resnet50(pretrained=True)
            self.feature_extractor.eval()
            self.feature_extractor.to(self.device)

            # Remove the final classification layer to get 2048-dim features
            self.feature_extractor = torch.nn.Sequential(
                *list(self.feature_extractor.children())[:-1]
            )

            # Define transform for feature extraction
            self.feature_extractor_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            Log("ResNet50 feature extractor loaded successfully", tag="PGO")
        except Exception as e:
            Log(f"Failed to load ResNet50 model: {e}", tag="PGO")
            self.feature_extractor = None

    def run(self):
        """Main loop of the PGO thread"""
        Log("PGO Thread started", tag="PGO")

        while True:
            try:
                # Check for new keyframes
                if not self.pgo_queue_in.empty():
                    data = self.pgo_queue_in.get()

                    if data[0] == "new_keyframe":
                        keyframe_data = data[1]
                        self._process_new_keyframe(keyframe_data)

                    elif data[0] == "optimize":
                        self._optimize_pose_graph()

                    elif data[0] == "stop":
                        Log("PGO Thread stopping", tag="PGO")
                        break

                time.sleep(0.01)  # Small delay to prevent busy waiting

            except Exception as e:
                Log(f"Error in PGO thread: {e}", tag="PGO")
                time.sleep(0.1)

    def _process_new_keyframe(self, keyframe_data):
        """Process a new keyframe for loop closure detection"""
        keyframe_id = keyframe_data["keyframe_id"]
        pose = keyframe_data["pose"]  # [R, t]
        rgb_image = keyframe_data["rgb_image"]  # HxWx3 numpy array
        timestamp = keyframe_data.get("timestamp", time.time())

        Log(f"Processing keyframe {keyframe_id}", tag="PGO")

        # Debug logging for image info
        if rgb_image is not None:
            Log(
                f"RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}, min: {rgb_image.min()}, max: {rgb_image.max()}",
                tag="PGO",
            )
        else:
            Log(f"RGB image is None for keyframe {keyframe_id}", tag="PGO")

        # Add new node to pose graph
        new_node = PoseGraphNode(keyframe_id, pose, timestamp)
        self.pose_graph_nodes[keyframe_id] = new_node

        # Add sequential edge if not the first keyframe
        if len(self.pose_graph_nodes) > 1:
            prev_keyframe_id = max(
                k for k in self.pose_graph_nodes.keys() if k < keyframe_id
            )
            if prev_keyframe_id is not None:
                prev_pose = self.pose_graph_nodes[prev_keyframe_id].pose
                relative_pose = self._compute_relative_pose(prev_pose, pose)

                sequential_edge = PoseGraphEdge(
                    from_id=prev_keyframe_id,
                    to_id=keyframe_id,
                    relative_pose=relative_pose,
                    edge_type="sequential",
                )
                self.pose_graph_edges.append(sequential_edge)

        # Detect landmarks and extract features
        landmarks = self._detect_landmarks(rgb_image, keyframe_id)
        new_node.landmarks = landmarks

        # Store landmarks in database
        for landmark in landmarks:
            landmark_id = f"{keyframe_id}_{len(self.landmark_db)}"
            self.landmark_db[landmark_id] = landmark

        # Check for loop closures
        loop_closure_found = self._detect_loop_closure(keyframe_id, landmarks)

        if loop_closure_found:
            self.loop_closure_count += 1
            Log(f"Loop closure detected! Total: {self.loop_closure_count}", tag="PGO")

            # Trigger optimization if needed
            if self.loop_closure_count % self.optimization_frequency == 0:
                self._optimize_pose_graph()

    def _detect_landmarks(self, rgb_image, keyframe_id):
        """Detect landmarks in the image using YOLOv8"""
        landmarks = []

        if self.yolo_model is None or self.feature_extractor is None:
            return landmarks

        try:
            # Validate input image
            if rgb_image is None or rgb_image.size == 0:
                Log(f"Invalid input image for keyframe {keyframe_id}", tag="PGO")
                return landmarks

            # Ensure image is in correct format (HxWx3)
            if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
                Log(
                    f"Invalid image shape: {rgb_image.shape} for keyframe {keyframe_id}",
                    tag="PGO",
                )
                return landmarks

            # Convert image to uint8 if needed (YOLOv8 expects uint8)
            if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
                if rgb_image.max() <= 1.0:
                    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
                else:
                    rgb_image_uint8 = rgb_image.astype(np.uint8)
                Log(
                    f"Converted image from {rgb_image.dtype} to uint8 for keyframe {keyframe_id}",
                    tag="PGO",
                )
            else:
                rgb_image_uint8 = rgb_image

            # Define static classes first
            static_classes = [
                # Cơ sở hạ tầng Giao thông (Rất tốt - Tĩnh, Bền vững, Dễ nhận dạng)
                "traffic light",  # Đèn giao thông
                "stop sign",  # Biển báo dừng
                "fire hydrant",  # Trụ cứu hỏa
                "street sign",  # Biển báo tên đường (nếu model của bạn nhận ra được)
                "parking meter",  # Đồng hồ đỗ xe
                "traffic sign",  # Các loại biển báo giao thông khác
                # Cấu trúc Xây dựng (Rất tốt - Tĩnh, Bền vững)
                "building",  # Tòa nhà
                "bridge",  # Cầu
                "tunnel",  # Hầm chui
                "fence",  # Hàng rào (tốt nếu có đặc điểm riêng)
                "wall",  # Tường (tốt nếu có graffiti, hoa văn, hoặc đặc điểm riêng)
                # Tiện ích Công cộng và Cảnh quan (Tốt - Tương đối tĩnh và bền vững)
                "bench",  # Ghế công cộng
                "pole",  # Cột điện, cột đèn (rất phổ biến)
                "utility pole",  # Cột điện
                "bus stop",  # Trạm xe buýt
                "billboard",  # Biển quảng cáo lớn
                # Thực vật (Có thể sử dụng nhưng cần cẩn thận)
                "tree",  # Cây (Chỉ nên dùng những cây lớn, đặc trưng. Cẩn thận vì hình dạng có thể thay đổi)
                "shrub",  # Bụi cây (kém tin cậy hơn cây)
                # Các thành phần kiến trúc chi tiết (Nếu segmentation model đủ tốt)
                "window",  # Cửa sổ (của tòa nhà)
                "door",  # Cửa ra vào (của tòa nhà)
                "roof",  # Mái nhà
                "balcony",  # Ban công
                "column",  # Cột nhà
                "staircase",  # Cầu thang bộ ngoài trời,
                "mailbox",      # Hộp thư
                "telephone booth",  # Buồng điện thoại
                "statue",       # Tượng đài
                "monument",     # Đài tưởng niệm
                "clock tower",  # Tháp đồng hồ
                "fountain",     # Đài phun nước
            ]

            # Run YOLOv8 detection first
            try:
                results = self.yolo_model(rgb_image_uint8)
                Log(f"YOLOv8 detection completed for keyframe {keyframe_id}", tag="PGO")
            except Exception as e:
                Log(f"Error in YOLOv8 detection: {e}", tag="PGO")
                return landmarks

            # Save debug image
            if self.save_dir:
                debug_dir = os.path.join(self.save_dir, "pgo_debug")
                os.makedirs(debug_dir, exist_ok=True)
                debug_image_path = os.path.join(
                    debug_dir, f"keyframe_{keyframe_id}_input.jpg"
                )
                cv2.imwrite(
                    debug_image_path, cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2BGR)
                )
                Log(f"Saved debug image: {debug_image_path}", tag="PGO")

                # Save image with detections for debugging
                debug_image_with_boxes = rgb_image_uint8.copy()
                for result in results:
                    if result.boxes is not None:
                        for i, (box, cls) in enumerate(
                            zip(result.boxes, result.boxes.cls)
                        ):
                            class_name = result.names[int(cls)]
                            confidence = (
                                float(result.boxes.conf[i])
                                if result.boxes.conf is not None
                                else 0.0
                            )
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # Draw bounding box
                            color = (
                                (0, 255, 0)
                                if class_name.lower() in static_classes
                                else (0, 0, 255)
                            )
                            cv2.rectangle(
                                debug_image_with_boxes, (x1, y1), (x2, y2), color, 2
                            )
                            cv2.putText(
                                debug_image_with_boxes,
                                f"{class_name} {confidence:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

                debug_boxes_path = os.path.join(
                    debug_dir, f"keyframe_{keyframe_id}_with_boxes.jpg"
                )
                cv2.imwrite(
                    debug_boxes_path,
                    cv2.cvtColor(debug_image_with_boxes, cv2.COLOR_RGB2BGR),
                )
                Log(f"Saved debug image with boxes: {debug_boxes_path}", tag="PGO")

            # Log all detections for debugging
            all_detections = []
            total_results = len(results)
            Log(
                f"YOLOv8 returned {total_results} results for keyframe {keyframe_id}",
                tag="PGO",
            )

            for i, result in enumerate(results):
                Log(
                    f"Processing result {i}: boxes={result.boxes is not None}, masks={result.masks is not None}",
                    tag="PGO",
                )
                if result.boxes is not None:
                    num_boxes = len(result.boxes)
                    Log(f"Result {i} has {num_boxes} boxes", tag="PGO")
                    for j, (box, cls) in enumerate(zip(result.boxes, result.boxes.cls)):
                        class_name = result.names[int(cls)]
                        confidence = (
                            float(result.boxes.conf[j])
                            if result.boxes.conf is not None
                            else 0.0
                        )
                        all_detections.append((class_name, confidence))
                        Log(
                            f"Box {j}: {class_name} (confidence: {confidence:.3f})",
                            tag="PGO",
                        )
                else:
                    Log(f"Result {i} has no boxes", tag="PGO")

            Log(
                f"YOLOv8 detected {len(all_detections)} objects in keyframe {keyframe_id}: {all_detections[:10]}",
                tag="PGO",
            )

            # Filter for static landmarks (exclude moving objects)
            Log(f"Looking for static classes: {static_classes[:10]}...", tag="PGO")

            landmark_count = 0
            for result in results:
                if result.boxes is not None and result.masks is not None:
                    for box, mask, cls in zip(
                        result.boxes, result.masks, result.boxes.cls
                    ):
                        try:
                            class_name = result.names[int(cls)]
                            confidence = (
                                float(
                                    result.boxes.conf[list(result.boxes.cls).index(cls)]
                                )
                                if result.boxes.conf is not None
                                else 0.0
                            )

                            # Log all detected classes
                            Log(
                                f"Detected object: {class_name} (confidence: {confidence:.3f})",
                                tag="PGO",
                            )

                            # Only consider static landmarks
                            if class_name.lower() in static_classes:
                                Log(
                                    f"Found static landmark: {class_name} (confidence: {confidence:.3f})",
                                    tag="PGO",
                                )
                                # Extract bounding box
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                bbox = [int(x1), int(y1), int(x2), int(y2)]

                                # Validate bounding box
                                if x1 >= x2 or y1 >= y2:
                                    continue

                                # Extract mask
                                mask_array = mask.data[0].cpu().numpy()

                                # Extract feature descriptor
                                Log(
                                    f"Calling _extract_feature_descriptor with image dtype: {rgb_image_uint8.dtype}",
                                    tag="PGO",
                                )
                                descriptor = self._extract_feature_descriptor(
                                    rgb_image_uint8, bbox
                                )

                                if descriptor is not None:
                                    landmark = Landmark(
                                        mask=mask_array,
                                        class_id=int(cls),
                                        class_name=class_name,
                                        bbox=bbox,
                                        descriptor=descriptor,
                                        keyframe_id=keyframe_id,
                                    )
                                    landmarks.append(landmark)
                                    landmark_count += 1

                        except Exception as e:
                            Log(f"Error processing individual landmark: {e}", tag="PGO")
                            continue

            Log(
                f"Detected {landmark_count} landmarks in keyframe {keyframe_id}",
                tag="PGO",
            )

        except Exception as e:
            Log(f"Error in landmark detection: {e}", tag="PGO")

        return landmarks

    def _extract_feature_descriptor(self, rgb_image, bbox):
        """Extract feature descriptor for a landmark using ResNet50"""
        try:
            Log(
                f"_extract_feature_descriptor called with image dtype: {rgb_image.dtype}, shape: {rgb_image.shape}",
                tag="PGO",
            )
            x1, y1, x2, y2 = bbox

            # Validate bounding box
            if x1 >= x2 or y1 >= y2:
                Log(f"Invalid bounding box: {bbox}", tag="PGO")
                return None

            # Ensure bounding box is within image bounds
            h, w = rgb_image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Crop the landmark region
            cropped_image = rgb_image[y1:y2, x1:x2]

            # Check if cropped image is valid
            if (
                cropped_image.size == 0
                or cropped_image.shape[0] == 0
                or cropped_image.shape[1] == 0
            ):
                Log(
                    f"Invalid cropped image size: {cropped_image.shape}, bbox: {bbox}",
                    tag="PGO",
                )
                return None

            # Ensure minimum size for feature extraction
            min_size = 10
            if cropped_image.shape[0] < min_size or cropped_image.shape[1] < min_size:
                Log(
                    f"Cropped image too small: {cropped_image.shape}, skipping",
                    tag="PGO",
                )
                return None

            # Convert to PIL and apply transform
            try:
                # Convert uint8 to PIL (rgb_image is now uint8)
                pil_image = transforms.ToPILImage()(cropped_image)
                transformed_image = self.feature_extractor_transform(pil_image)

                # Add batch dimension
                transformed_image = transformed_image.unsqueeze(0).to(self.device)

                # Extract features
                with torch.no_grad():
                    features = self.feature_extractor(transformed_image)
                    descriptor = features.squeeze().cpu().numpy()

                Log(
                    f"Successfully extracted descriptor for bbox {bbox}, shape: {descriptor.shape}",
                    tag="PGO",
                )
                return descriptor

            except Exception as e:
                Log(
                    f"Error in PIL/transform processing: {e}, cropped_image shape: {cropped_image.shape}, dtype: {cropped_image.dtype}",
                    tag="PGO",
                )
                return None

        except Exception as e:
            Log(f"Error extracting feature descriptor: {e}, bbox: {bbox}", tag="PGO")
            return None

    def _detect_loop_closure(self, current_keyframe_id, current_landmarks):
        """Detect loop closure by comparing landmarks with previous keyframes"""
        if len(current_landmarks) == 0:
            return False

        # Find candidate keyframes based on landmark similarity
        keyframe_votes = defaultdict(int)

        for landmark in current_landmarks:
            # Find similar landmarks in database
            similar_landmarks = self._find_similar_landmarks(landmark)

            # Vote for keyframes that contain similar landmarks
            for similar_landmark in similar_landmarks:
                if similar_landmark.keyframe_id != current_keyframe_id:
                    # Check minimum distance requirement
                    if (
                        abs(similar_landmark.keyframe_id - current_keyframe_id)
                        >= self.min_loop_closure_distance
                    ):
                        keyframe_votes[similar_landmark.keyframe_id] += 1

        # Find the keyframe with most votes
        if keyframe_votes:
            best_candidate_id = max(keyframe_votes.items(), key=lambda x: x[1])[0]

            # Check if we have enough votes
            if keyframe_votes[best_candidate_id] >= 2:  # At least 2 matching landmarks
                # Perform geometric verification
                if self._geometric_verification(current_keyframe_id, best_candidate_id):
                    # Add loop closure edge
                    self._add_loop_closure_edge(current_keyframe_id, best_candidate_id)
                    return True

        return False

    def _find_similar_landmarks(self, query_landmark):
        """Find landmarks similar to the query landmark based on feature similarity"""
        similar_landmarks = []

        for landmark_id, landmark in self.landmark_db.items():
            if landmark.keyframe_id == query_landmark.keyframe_id:
                continue

            # Compute cosine similarity
            similarity = self._compute_cosine_similarity(
                query_landmark.descriptor, landmark.descriptor
            )

            if similarity > self.loop_closure_threshold:
                similar_landmarks.append(landmark)

        # Sort by similarity and return top candidates
        similar_landmarks.sort(
            key=lambda x: self._compute_cosine_similarity(
                query_landmark.descriptor, x.descriptor
            ),
            reverse=True,
        )

        return similar_landmarks[: self.max_loop_closure_candidates]

    def _compute_cosine_similarity(self, desc1, desc2):
        """Compute cosine similarity between two descriptors"""
        try:
            # Normalize descriptors
            desc1_norm = desc1 / (np.linalg.norm(desc1) + 1e-8)
            desc2_norm = desc2 / (np.linalg.norm(desc2) + 1e-8)

            # Compute cosine similarity
            similarity = np.dot(desc1_norm, desc2_norm)
            return similarity
        except:
            return 0.0

    def _geometric_verification(self, kf1_id, kf2_id):
        """Perform geometric verification between two keyframes using ICP"""
        try:
            # Get poses
            pose1 = self.pose_graph_nodes[kf1_id].pose
            pose2 = self.pose_graph_nodes[kf2_id].pose

            # For now, use a more reasonable geometric verification
            # In a full implementation, you would:
            # 1. Get point clouds from both keyframes
            # 2. Use ICP to align them
            # 3. Check alignment quality

            # Current implementation: Check if poses are reasonable for loop closure
            t1, t2 = pose1[1], pose2[1]
            distance = np.linalg.norm(t1 - t2)

            # Loop closure should have reasonable distance (not too close, not too far)
            # Too close (< 2m): might be same location, no loop closure needed
            # Too far (> 50m): unlikely to be same location
            # Sweet spot: 5-30m for urban environments
            min_distance = 2.0  # meters
            max_distance = 50.0  # meters

            if min_distance <= distance <= max_distance:
                # Additional check: relative orientation should be reasonable
                R1, R2 = pose1[0], pose2[0]
                relative_rotation = R2 @ R1.T

                # Convert to Euler angles for easier interpretation
                # Check if rotation is reasonable (not too extreme)
                # This is a simplified check - in practice you'd use more sophisticated methods

                Log(f"Geometric verification: distance={distance:.2f}m, kf1={kf1_id}, kf2={kf2_id}", tag="PGO")
                return True
            else:
                Log(f"Geometric verification failed: distance={distance:.2f}m outside range [{min_distance}, {max_distance}]", tag="PGO")
                return False

        except Exception as e:
            Log(f"Error in geometric verification: {e}", tag="PGO")
            return False

    def _add_loop_closure_edge(self, kf1_id, kf2_id):
        """Add a loop closure edge to the pose graph"""
        pose1 = self.pose_graph_nodes[kf1_id].pose
        pose2 = self.pose_graph_nodes[kf2_id].pose

        # Compute relative pose (simplified)
        relative_pose = self._compute_relative_pose(pose1, pose2)

        # Assign higher information matrix for loop closure edges
        # Loop closure edges are more reliable than sequential edges
        # Use higher weights for rotation and translation
        loop_closure_info_matrix = np.eye(6) * 10.0  # 10x higher confidence than sequential

        loop_closure_edge = PoseGraphEdge(
            from_id=kf1_id,
            to_id=kf2_id,
            relative_pose=relative_pose,
            edge_type="loop_closure",
            information_matrix=loop_closure_info_matrix,
        )

        self.pose_graph_edges.append(loop_closure_edge)
        Log(
            f"Added loop closure edge between keyframes {kf1_id} and {kf2_id}",
            tag="PGO",
        )

    def _compute_relative_pose(self, pose1, pose2):
        """Compute relative pose from pose1 to pose2"""
        R1, t1 = pose1
        R2, t2 = pose2

        # Relative rotation: R_rel = R2 * R1^T
        R_rel = R2 @ R1.T

        # Relative translation: t_rel = R2 * (-R1^T * t1) + t2
        t_rel = R2 @ (-R1.T @ t1) + t2

        return [R_rel, t_rel]

    def _optimize_pose_graph(self):
        """Optimize the pose graph using g2o"""
        if not G2O_AVAILABLE:
            Log("g2o not available, skipping pose graph optimization", tag="PGO")
            return

        if len(self.pose_graph_nodes) < 2:
            return

        try:
            # Create g2o optimizer
            optimizer = g2o.SparseOptimizer()
            solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
            solver = g2o.OptimizationAlgorithmLevenberg(solver)
            optimizer.set_algorithm(solver)

            # Add vertices (poses)
            vertices = {}
            for keyframe_id, node in self.pose_graph_nodes.items():
                pose = node.pose
                R, t = pose

                # Convert to g2o format
                g2o_pose = g2o.SE3Quat(R, t)
                vertex = g2o.VertexSE3()
                vertex.set_id(keyframe_id)
                vertex.set_estimate(g2o_pose)

                # Fix the first vertex
                if keyframe_id == min(self.pose_graph_nodes.keys()):
                    vertex.set_fixed(True)

                optimizer.add_vertex(vertex)
                vertices[keyframe_id] = vertex

            # Add edges (constraints)
            for edge in self.pose_graph_edges:
                R_rel, t_rel = edge.relative_pose
                g2o_pose = g2o.SE3Quat(R_rel, t_rel)

                edge_g2o = g2o.EdgeSE3()
                edge_g2o.set_vertex(0, vertices[edge.from_id])
                edge_g2o.set_vertex(1, vertices[edge.to_id])
                edge_g2o.set_measurement(g2o_pose)
                edge_g2o.set_information(edge.information_matrix)

                optimizer.add_edge(edge_g2o)

            # Optimize
            optimizer.initialize_optimization()
            optimizer.optimize(20)  # 20 iterations

            # Extract optimized poses
            optimized_poses = {}
            for keyframe_id, vertex in vertices.items():
                optimized_pose = vertex.estimate()
                R_opt = optimized_pose.rotation().matrix()
                t_opt = optimized_pose.translation()
                optimized_poses[keyframe_id] = [R_opt, t_opt]

            # Send optimized poses to backend
            self.pgo_queue_out.put(["optimized_poses", optimized_poses])

            Log(
                f"Pose graph optimization completed for {len(optimized_poses)} keyframes",
                tag="PGO",
            )

        except Exception as e:
            Log(f"Error in pose graph optimization: {e}", tag="PGO")
