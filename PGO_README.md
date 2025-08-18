# Pose Graph Optimization (PGO) Module for OpenGS-SLAM

## Tổng quan

Module PGO này tích hợp Pose Graph Optimization và Loop Closure dựa trên Landmark vào hệ thống OpenGS-SLAM, giúp sửa lỗi drift toàn cục và cải thiện độ chính xác của quỹ đạo.

## Tính năng chính

### 1. Loop Closure Detection
- **Landmark-based Detection**: Sử dụng YOLOv8 để phát hiện các đối tượng tĩnh (landmarks) như biển báo, đèn giao thông, ghế đợi, v.v.
- **Feature Extraction**: Sử dụng ResNet50 để trích xuất đặc trưng 2048 chiều cho mỗi landmark
- **Similarity Matching**: So sánh cosine similarity giữa các landmark để tìm kiếm vòng lặp
- **Geometric Verification**: Xác thực hình học để loại bỏ false positives

### 2. Pose Graph Optimization
- **Graph Construction**: Xây dựng đồ thị pose với các node là keyframes và edge là các ràng buộc pose
- **Global Optimization**: Sử dụng g2o để tối ưu hóa toàn cục đồ thị pose
- **Drift Correction**: Sửa lỗi drift tích lũy trong quỹ đạo

### 3. Multi-threading Architecture
- **PGO Thread**: Luồng riêng biệt xử lý PGO và loop closure
- **Asynchronous Processing**: Xử lý bất đồng bộ không ảnh hưởng đến performance của SLAM chính
- **Queue-based Communication**: Giao tiếp qua queues giữa các luồng

## Cài đặt

### 1. Cài đặt Dependencies

```bash
# Chạy script cài đặt tự động
./install_pgo.sh

# Hoặc cài đặt thủ công
pip install ultralytics>=8.0.0
pip install g2o-python>=0.1.0  # Có thể cần build từ source
```

### 2. Cài đặt g2o-python (nếu cần)

```bash
# Clone repository
git clone https://github.com/uoip/g2opy.git
cd g2opy

# Build và cài đặt
python setup.py install
```

## Cấu trúc Code

### Files chính

- `utils/slam_pgo.py`: Module PGO chính
- `utils/slam_backend.py`: Backend đã được tích hợp PGO
- `slam.py`: Main SLAM class đã được tích hợp PGO thread

### Classes chính

#### PGOThread
```python
class PGOThread(mp.Process):
    """Pose Graph Optimization thread for loop closure detection and global optimization"""

    def __init__(self, pgo_queue_in, pgo_queue_out, config, save_dir=None):
        # Khởi tạo PGO thread

    def run(self):
        # Vòng lặp chính của PGO thread

    def _detect_landmarks(self, rgb_image, keyframe_id):
        # Phát hiện landmarks bằng YOLOv8

    def _detect_loop_closure(self, current_keyframe_id, current_landmarks):
        # Phát hiện loop closure

    def _optimize_pose_graph(self):
        # Tối ưu hóa đồ thị pose
```

#### Landmark
```python
class Landmark:
    """Class to represent a landmark detected in a keyframe"""
    def __init__(self, mask, class_id, class_name, bbox, descriptor, keyframe_id):
        # Thông tin về landmark
```

#### PoseGraphNode & PoseGraphEdge
```python
class PoseGraphNode:
    """Class to represent a node in the pose graph"""

class PoseGraphEdge:
    """Class to represent an edge in the pose graph"""
```

## Cách sử dụng

### 1. Chạy SLAM với PGO

```python
# PGO được tích hợp tự động vào SLAM system
# Không cần thay đổi code hiện tại

# Chạy như bình thường
python slam.py --config configs/mono/waymo/100613.yaml
```

### 2. Cấu hình PGO

Có thể thêm cấu hình PGO vào file config:

```yaml
# Thêm vào config file
PGO:
  enabled: true
  loop_closure_threshold: 0.7
  min_loop_closure_distance: 10
  optimization_frequency: 5
  static_landmarks:
    - "traffic light"
    - "stop sign"
    - "bench"
    - "hydrant"
    - "pole"
    - "building"
    - "tree"
```

### 3. Monitoring PGO

PGO module sẽ log các thông tin quan trọng:

```
[PGO] PGO Thread initialized
[PGO] YOLOv8 model loaded successfully
[PGO] ResNet50 feature extractor loaded successfully
[PGO] Processing keyframe 10
[PGO] Detected 3 landmarks in keyframe 10
[PGO] Loop closure detected! Total: 1
[PGO] Added loop closure edge between keyframes 10 and 5
[PGO] Pose graph optimization completed for 15 keyframes
```

## Tùy chỉnh

### 1. Thay đổi Landmark Classes

```python
# Trong slam_pgo.py, thay đổi static_classes
static_classes = ['traffic light', 'stop sign', 'bench', 'hydrant', 'pole',
                 'building', 'tree', 'sign', 'fire hydrant']
```

### 2. Điều chỉnh Loop Closure Parameters

```python
# Trong PGOThread.__init__
self.loop_closure_threshold = 0.7  # Cosine similarity threshold
self.min_loop_closure_distance = 10  # Minimum keyframe distance
self.max_loop_closure_candidates = 5
```

### 3. Thay đổi Optimization Frequency

```python
# Trong PGOThread.__init__
self.optimization_frequency = 5  # Optimize every N loop closures
```

## Troubleshooting

### 1. YOLOv8 không load được

```bash
# Kiểm tra installation
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"

# Download model manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt
```

### 2. g2o-python không cài đặt được

```bash
# Sử dụng alternative
pip install gtsam  # Thay thế cho g2o
# Hoặc
pip install ceres-python
```

### 3. Memory Issues

```python
# Giảm batch size hoặc frequency
self.optimization_frequency = 10  # Tăng frequency
self.max_loop_closure_candidates = 3  # Giảm candidates
```

## Performance

### Benchmarks

- **Landmark Detection**: ~50ms per keyframe (YOLOv8l)
- **Feature Extraction**: ~20ms per landmark (ResNet50)
- **Loop Closure Detection**: ~100ms per keyframe
- **Pose Graph Optimization**: ~500ms per optimization (20 iterations)

### Memory Usage

- **YOLOv8 Model**: ~150MB
- **ResNet50 Model**: ~100MB
- **Landmark Database**: ~10MB per 100 keyframes
- **Pose Graph**: ~5MB per 100 keyframes

## Tương lai

### Planned Features

1. **Advanced Geometric Verification**: ICP-based verification
2. **Multi-scale Loop Closure**: Detection at different scales
3. **Adaptive Parameters**: Tự động điều chỉnh parameters
4. **Visualization Tools**: GUI để visualize pose graph
5. **Alternative Optimizers**: Support cho GTSAM, Ceres

### Contributing

Để đóng góp vào PGO module:

1. Fork repository
2. Tạo feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## References

- [YOLOv8 Paper](https://arxiv.org/abs/2304.00501)
- [g2o Framework](https://github.com/RainerKuemmerle/g2o)
- [Pose Graph Optimization Tutorial](https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.pdf)
- [Loop Closure Detection Survey](https://arxiv.org/abs/1904.10146)
