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
        self.edge_type = edge_type  # 'sequential' or 'place_recognition'
        self.information_matrix = (
            information_matrix if information_matrix is not None else np.eye(6)
        )


class PGOThread(mp.Process):
    """Pose Graph Optimization thread for place recognition and global optimization

    This implementation is specifically designed for Waymo dataset which lacks true loop closures.
    Instead of traditional loop closure detection, we use place recognition to detect when
    the same landmarks are observed from different viewpoints (e.g., parallel roads,
    intersections, different angles of the same building). This helps correct drift and
    maintain global consistency of the map.
    """

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

        # Place recognition parameters (adapted for Waymo dataset)
        self.place_recognition_threshold = 0.7  # Reduced from 0.8
        self.max_place_recognition_candidates = 5
        self.min_place_recognition_distance = 5  # Reduced from 10

        # Optimization parameters
        self.optimization_frequency = 5  # Optimize every N place recognitions
        self.place_recognition_count = 0
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
        """Process a new keyframe for place recognition detection"""
        keyframe_id = keyframe_data["keyframe_id"]
        pose = keyframe_data["pose"]  # [R, t]
        rgb_image = keyframe_data["rgb_image"]  # HxWx3 numpy array
        timestamp = keyframe_data.get("timestamp", time.time())

        Log(f"Processing keyframe {keyframe_id}", tag="PGO")

        # Debug logging for pose scale
        R, t = pose
        Log(f"Pose scale debug - keyframe {keyframe_id}: R shape={R.shape}, t shape={t.shape}, t values={t}, t norm={np.linalg.norm(t):.4f}", tag="PGO")

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

        # Check for place recognition (not loop closure)
        place_recognition_found = self._detect_place_recognition()

        if place_recognition_found:
            self.place_recognition_count += 1
            Log(f"Place recognition detected! Total: {self.place_recognition_count}", tag="PGO")

            # Trigger optimization if needed
            if self.place_recognition_count % self.optimization_frequency == 0:
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
    # === CẤP 1: Landmark Siêu Ổn định (High-Confidence Static Landmarks) ===
    # Đây là những đối tượng gần như chắc chắn không di chuyển.
    'traffic light',
    'stop sign',
    'fire hydrant',
    'bench',
    'parking meter', # Có trong COCO, nhưng có thể ít xuất hiện

    # === CẤP 2: Landmark Tĩnh Hiệu quả (Effectively Static Landmarks) ===
    # Đây là những đối tượng tĩnh trong phần lớn các sequence ngắn của Waymo.
    # Chúng ta sẽ sử dụng chúng nhưng có thể áp dụng một trọng số tin cậy thấp hơn trong PGO.
    'car',
    'truck',
    'bus',

    # === CẤP 3: Landmark Môi trường (Environmental Landmarks) ===
    # Những đối tượng này cũng tĩnh, nhưng có thể khó trích xuất đặc trưng một cách nhất quán.
    'potted plant', # Thường thấy trước các tòa nhà, cửa hàng
    'backpack', 'suitcase', 'handbag' # Đôi khi bị bỏ lại và trở thành tĩnh, nhưng rủi ro cao
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

    def _detect_place_recognition(self):
        """Detect place recognition opportunities using landmark matching"""
        try:
            if len(self.landmark_db) < 2:
                Log(f"PGO: Not enough landmarks for place recognition ({len(self.landmark_db)})", tag="PGO")
                return False

            Log(f"PGO: Starting place recognition with {len(self.landmark_db)} landmarks", tag="PGO")

            # Get current keyframe ID
            current_keyframe_id = max(self.pose_graph_nodes.keys())
            current_landmarks = [lm for lm in self.landmark_db.values() if lm.keyframe_id == current_keyframe_id]

            Log(f"PGO: Current keyframe {current_keyframe_id} has {len(current_landmarks)} landmarks", tag="PGO")

            if len(current_landmarks) == 0:
                Log(f"PGO: No landmarks in current keyframe {current_keyframe_id}", tag="PGO")
                return False

            # Initialize voting system
            keyframe_votes = defaultdict(int)
            landmark_matches = defaultdict(list)  # Store actual landmark matches

            for landmark in current_landmarks:
                # Find similar landmarks in database
                similar_landmarks = self._find_similar_landmarks(landmark)
                Log(f"PGO: Landmark {landmark.class_name} has {len(similar_landmarks)} similar landmarks", tag="PGO")

                # Vote for keyframes that contain similar landmarks
                for similar_landmark in similar_landmarks:
                    if similar_landmark.keyframe_id != current_keyframe_id:
                        # Check minimum distance requirement (but not too strict)
                        # For place recognition, we want to detect when we see the same place
                        # from a different viewpoint, even if not a true loop closure
                        distance = abs(similar_landmark.keyframe_id - current_keyframe_id)

                        # Temporal filtering: avoid detecting place recognition between very close keyframes
                        # This prevents false positives when the vehicle is stationary or moving very slowly
                        min_temporal_distance = self.min_place_recognition_distance  # Use configurable value
                        if distance >= min_temporal_distance:
                            keyframe_votes[similar_landmark.keyframe_id] += 1
                            landmark_matches[similar_landmark.keyframe_id].append(
                                (landmark, similar_landmark)
                            )
                            Log(f"PGO: Vote for keyframe {similar_landmark.keyframe_id} (distance={distance})", tag="PGO")

            Log(f"PGO: Keyframe votes: {dict(keyframe_votes)}", tag="PGO")

            # Find the keyframe with most votes
            if keyframe_votes:
                best_candidate_id = max(keyframe_votes.items(), key=lambda x: x[1])[0]
                vote_count = keyframe_votes[best_candidate_id]

                Log(f"PGO: Best candidate keyframe {best_candidate_id} with {vote_count} votes", tag="PGO")

                # For place recognition, we can be more lenient than strict loop closure
                # We want to detect when we see the same place from different viewpoints
                if vote_count >= 1:  # At least 1 matching landmark
                    # Perform geometric verification for place recognition
                    if self._verify_place_recognition(current_keyframe_id, best_candidate_id, landmark_matches[best_candidate_id]):
                        # Add place recognition constraint (not loop closure)
                        self._add_place_recognition_constraint(current_keyframe_id, best_candidate_id, landmark_matches[best_candidate_id])
                        Log(f"PGO: Successfully added place recognition edge between keyframes {current_keyframe_id} and {best_candidate_id}", tag="PGO")
                        return True
                    else:
                        Log(f"PGO: Place recognition verification failed for keyframes {current_keyframe_id} and {best_candidate_id}", tag="PGO")
                else:
                    Log(f"PGO: Insufficient votes ({vote_count}) for place recognition", tag="PGO")
            else:
                Log(f"PGO: No keyframe votes for place recognition", tag="PGO")

            # FALLBACK: Force create a place recognition edge for testing
            # This helps us verify that PGO optimization works even without natural place recognition
            if len(self.pose_graph_nodes) >= 10:  # Only if we have enough keyframes
                # Find a keyframe that's far enough back
                candidate_keyframes = [kf_id for kf_id in self.pose_graph_nodes.keys()
                                     if abs(kf_id - current_keyframe_id) >= 5 and kf_id != current_keyframe_id]

                if candidate_keyframes:
                    # Pick the keyframe that's furthest back
                    test_keyframe_id = min(candidate_keyframes)
                    Log(f"PGO: FALLBACK - Creating test place recognition edge between keyframes {current_keyframe_id} and {test_keyframe_id}", tag="PGO")

                    # Create a dummy landmark match for testing
                    dummy_matches = []
                    self._add_place_recognition_constraint(current_keyframe_id, test_keyframe_id, dummy_matches)
                    return True

            return False

        except Exception as e:
            Log(f"Error in place recognition detection: {e}", tag="PGO")
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

            if similarity > self.place_recognition_threshold:
                similar_landmarks.append(landmark)

        # Sort by similarity and return top candidates
        similar_landmarks.sort(
            key=lambda x: self._compute_cosine_similarity(
                query_landmark.descriptor, x.descriptor
            ),
            reverse=True,
        )

        return similar_landmarks[: self.max_place_recognition_candidates]

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


    def _add_place_recognition_constraint(self, kf1_id, kf2_id, landmark_matches):
        """Add a place recognition constraint to the pose graph"""
        pose1 = self.pose_graph_nodes[kf1_id].pose
        pose2 = self.pose_graph_nodes[kf2_id].pose

        # Compute relative pose
        relative_pose = self._compute_relative_pose(pose1, pose2)

        # For place recognition constraints, we use moderate information matrix
        # These are less certain than true loop closures but still valuable
        place_recognition_info_matrix = np.eye(6) * 5.0  # Moderate confidence

        place_recognition_edge = PoseGraphEdge(
            from_id=kf1_id,
            to_id=kf2_id,
            relative_pose=relative_pose,
            edge_type="place_recognition",  # New edge type
            information_matrix=place_recognition_info_matrix,
        )

        self.pose_graph_edges.append(place_recognition_edge)
        Log(
            f"Added place recognition constraint between keyframes {kf1_id} and {kf2_id} with {len(landmark_matches)} landmark matches",
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
        try:
            if len(self.pose_graph_nodes) < 2:
                Log(f"PGO: Not enough nodes for optimization ({len(self.pose_graph_nodes)})", tag="PGO")
                return

            Log(f"PGO: Starting pose graph optimization with {len(self.pose_graph_nodes)} nodes and {len(self.pose_graph_edges)} edges", tag="PGO")

            # Count edge types for debugging
            sequential_edges = sum(1 for edge in self.pose_graph_edges if edge.edge_type == "sequential")
            place_recognition_edges = sum(1 for edge in self.pose_graph_edges if edge.edge_type == "place_recognition")
            Log(f"PGO: Edge breakdown - Sequential: {sequential_edges}, Place Recognition: {place_recognition_edges}", tag="PGO")

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
                g2o_pose = g2o.SE3Quat(R, t)
                vertex = g2o.VertexSE3Expmap()  # Use VertexSE3Expmap for SE3Quat compatibility
                vertex.set_id(keyframe_id)
                vertex.set_estimate(g2o_pose)

                # Fix the first vertex to avoid gauge freedom
                if keyframe_id == min(self.pose_graph_nodes.keys()):
                    vertex.set_fixed(True)
                    Log(f"PGO: Fixed vertex {keyframe_id} to avoid gauge freedom", tag="PGO")

                optimizer.add_vertex(vertex)
                vertices[keyframe_id] = vertex

            # Add edges (constraints)
            for edge in self.pose_graph_edges:
                R_rel, t_rel = edge.relative_pose
                g2o_pose = g2o.SE3Quat(R_rel, t_rel)

                edge_g2o = g2o.EdgeSE3Expmap()  # Use EdgeSE3Expmap for SE3Quat compatibility
                edge_g2o.set_vertex(0, vertices[edge.from_id])
                edge_g2o.set_vertex(1, vertices[edge.to_id])
                edge_g2o.set_measurement(g2o_pose)

                # Set information matrix based on edge type
                if edge.edge_type == "sequential":
                    # Sequential edges have standard confidence
                    info_matrix = edge.information_matrix
                    Log(f"PGO: Sequential edge {edge.from_id}->{edge.to_id}, weight: 1.0", tag="PGO")
                elif edge.edge_type == "place_recognition":
                    # Place recognition edges have moderate confidence
                    info_matrix = edge.information_matrix * 2.0  # Reduced from 5.0
                    Log(f"PGO: Place recognition edge {edge.from_id}->{edge.to_id}, weight: 2.0", tag="PGO")
                else:
                    info_matrix = edge.information_matrix
                    Log(f"PGO: Unknown edge type {edge.edge_type}, weight: 1.0", tag="PGO")

                edge_g2o.set_information(info_matrix)
                optimizer.add_edge(edge_g2o)

            # Debug: Log initial state
            Log(f"PGO: Starting optimization with {len(vertices)} vertices and {len(self.pose_graph_edges)} edges", tag="PGO")

            # Optimize
            optimizer.initialize_optimization()
            optimizer.optimize(50)  # 50 iterations

            # Debug: Check optimization status
            active_edges = optimizer.active_edges()
            Log(f"PGO: Optimization completed. Active edges: {len(active_edges)}", tag="PGO")
            if len(active_edges) == 0:
                Log(f"PGO: Warning - No active edges after optimization", tag="PGO")

            # Extract optimized poses
            optimized_poses = {}
            pose_changes = []
            for keyframe_id, vertex in vertices.items():
                optimized_pose = vertex.estimate()
                R_opt = optimized_pose.rotation().matrix()
                t_opt = optimized_pose.translation()
                optimized_poses[keyframe_id] = [R_opt, t_opt]

                # Track pose changes for debugging
                if keyframe_id in self.pose_graph_nodes:
                    original_pose = self.pose_graph_nodes[keyframe_id].pose
                    R_orig, t_orig = original_pose
                    t_change = np.linalg.norm(t_opt - t_orig)
                    pose_changes.append(t_change)

            # Log pose change statistics
            if pose_changes:
                avg_change = np.mean(pose_changes)
                max_change = np.max(pose_changes)
                Log(f"PGO: Pose changes - avg: {avg_change:.4f}m, max: {max_change:.4f}m", tag="PGO")

            # Send optimized poses to backend
            self.pgo_queue_out.put(["optimized_poses", optimized_poses])

            Log(
                f"Pose graph optimization completed for {len(optimized_poses)} keyframes",
                tag="PGO",
            )

        except Exception as e:
            Log(f"Error in pose graph optimization: {e}", tag="PGO")

    def _verify_place_recognition(self, kf1_id, kf2_id, landmark_matches):
        """Verify place recognition between two keyframes using landmark matches"""
        try:
            # Get poses
            pose1 = self.pose_graph_nodes[kf1_id].pose
            pose2 = self.pose_graph_nodes[kf2_id].pose

            # For place recognition, we want to verify that:
            # 1. The landmarks are observed from reasonable viewpoints
            # 2. The relative pose between keyframes is geometrically consistent

            t1, t2 = pose1[1], pose2[1]
            distance = np.linalg.norm(t1 - t2)

            # Calculate temporal distance between keyframes
            temporal_distance = abs(kf1_id - kf2_id)

            # Debug logging to understand pose scale
            Log(f"Debug poses - kf1={kf1_id}: t1={t1}, kf2={kf2_id}: t2={t2}, distance={distance:.4f}", tag="PGO")

            # For place recognition, we're more lenient than strict loop closure
            # We want to detect when we see the same place from different viewpoints
            # This could happen when:
            # - Driving on parallel roads
            # - Crossing intersections
            # - Seeing the same building from different angles

            # Adjust distance range based on observed scale
            # If distances are in [0-1] range, this suggests poses might be normalized
            # For normalized poses, we need much smaller thresholds
            if distance < 1.0:
                # Poses appear to be normalized or in small scale
                min_distance = 0.1  # meters - increased minimum to avoid false positives
                max_distance = 5.0  # meters - reduced maximum for normalized poses
                Log(f"Using small scale distance range: [{min_distance}, {max_distance}]", tag="PGO")
            else:
                # Poses appear to be in normal scale
                min_distance = 2.0  # meters - increased minimum for normal scale
                max_distance = 100.0  # meters - maximum reasonable distance for place recognition
                Log(f"Using normal scale distance range: [{min_distance}, {max_distance}]", tag="PGO")

            # Additional checks for place recognition validity
            num_matches = len(landmark_matches)

            # Check if we have enough landmark matches relative to the distance
            # For small distances, we need more matches to be confident
            if distance < 1.0:
                min_matches = 3  # Need more matches for small distances
            else:
                min_matches = 1  # Fewer matches okay for larger distances

            if min_distance <= distance <= max_distance and num_matches >= min_matches:
                Log(f"Place recognition verified: distance={distance:.4f}m, matches={num_matches}, kf1={kf1_id}, kf2={kf2_id}", tag="PGO")
                return True
            else:
                if distance < min_distance or distance > max_distance:
                    Log(f"Place recognition failed: distance={distance:.4f}m outside range [{min_distance}, {max_distance}]", tag="PGO")
                if num_matches < min_matches:
                    Log(f"Place recognition failed: insufficient landmark matches ({num_matches} < {min_matches})", tag="PGO")
                return False

        except Exception as e:
            Log(f"Error in place recognition verification: {e}", tag="PGO")
            return False
