#!/usr/bin/env python3
"""
Test script for PGO (Pose Graph Optimization) module
This script tests the basic functionality of the PGO module
"""

import numpy as np
import torch
import time
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pgo_import():
    """Test if PGO module can be imported"""
    try:
        from utils.slam_pgo import PGOThread, Landmark, PoseGraphNode, PoseGraphEdge
        print("✓ PGO module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import PGO module: {e}")
        return False

def test_yolo_import():
    """Test if YOLOv8 can be imported"""
    try:
        from ultralytics import YOLO
        print("✓ YOLOv8 imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import YOLOv8: {e}")
        print("  Install with: pip install ultralytics")
        return False

def test_g2o_import():
    """Test if g2o can be imported"""
    try:
        import g2o
        print("✓ g2o imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import g2o: {e}")
        print("  Install with: pip install g2o-python")
        print("  Or build from source: https://github.com/uoip/g2opy")
        return False

def test_landmark_class():
    """Test Landmark class functionality"""
    try:
        from utils.slam_pgo import Landmark

        # Create a dummy landmark
        mask = np.random.rand(100, 100) > 0.5
        bbox = [10, 20, 100, 150]
        descriptor = np.random.rand(2048)

        landmark = Landmark(
            mask=mask,
            class_id=0,
            class_name="traffic light",
            bbox=bbox,
            descriptor=descriptor,
            keyframe_id=1
        )

        assert landmark.class_name == "traffic light"
        assert landmark.keyframe_id == 1
        assert landmark.descriptor.shape == (2048,)

        print("✓ Landmark class works correctly")
        return True
    except Exception as e:
        print(f"✗ Landmark class test failed: {e}")
        return False

def test_pose_graph_classes():
    """Test PoseGraphNode and PoseGraphEdge classes"""
    try:
        from utils.slam_pgo import PoseGraphNode, PoseGraphEdge

        # Test PoseGraphNode
        R = np.eye(3)
        t = np.array([1, 2, 3])
        pose = [R, t]

        node = PoseGraphNode(keyframe_id=1, pose=pose, timestamp=time.time())
        assert node.keyframe_id == 1
        assert len(node.landmarks) == 0

        # Test PoseGraphEdge
        edge = PoseGraphEdge(
            from_id=1,
            to_id=2,
            relative_pose=pose,
            edge_type='sequential'
        )
        assert edge.from_id == 1
        assert edge.to_id == 2
        assert edge.edge_type == 'sequential'

        print("✓ PoseGraph classes work correctly")
        return True
    except Exception as e:
        print(f"✗ PoseGraph classes test failed: {e}")
        return False

def test_cosine_similarity():
    """Test cosine similarity computation"""
    try:
        from utils.slam_pgo import PGOThread

        # Create a dummy PGOThread instance
        import torch.multiprocessing as mp
        pgo_queue_in = mp.Queue()
        pgo_queue_out = mp.Queue()

        # Mock config
        config = {"Training": {"monocular": True}}

        pgo = PGOThread(pgo_queue_in, pgo_queue_out, config)

        # Test cosine similarity
        desc1 = np.random.rand(2048)
        desc2 = np.random.rand(2048)

        similarity = pgo._compute_cosine_similarity(desc1, desc2)

        assert 0 <= similarity <= 1  # Cosine similarity should be in [0, 1]

        # Test with identical vectors
        similarity_identical = pgo._compute_cosine_similarity(desc1, desc1)
        assert abs(similarity_identical - 1.0) < 1e-6

        print("✓ Cosine similarity computation works correctly")
        return True
    except Exception as e:
        print(f"✗ Cosine similarity test failed: {e}")
        return False

def test_relative_pose_computation():
    """Test relative pose computation"""
    try:
        from utils.slam_pgo import PGOThread
        import torch.multiprocessing as mp

        pgo_queue_in = mp.Queue()
        pgo_queue_out = mp.Queue()
        config = {"Training": {"monocular": True}}

        pgo = PGOThread(pgo_queue_in, pgo_queue_out, config)

        # Test poses
        R1 = np.eye(3)
        t1 = np.array([0, 0, 0])
        pose1 = [R1, t1]

        R2 = np.eye(3)
        t2 = np.array([1, 0, 0])
        pose2 = [R2, t2]

        # Compute relative pose
        R_rel, t_rel = pgo._compute_relative_pose(pose1, pose2)

        # Check that relative translation is correct
        assert np.allclose(t_rel, np.array([1, 0, 0]), atol=1e-6)

        print("✓ Relative pose computation works correctly")
        return True
    except Exception as e:
        print(f"✗ Relative pose computation test failed: {e}")
        return False

def test_yolo_model_loading():
    """Test YOLOv8 model loading (if available)"""
    try:
        from ultralytics import YOLO

        # Try to load the model
        model = YOLO('yolov8l-seg.pt')
        print("✓ YOLOv8 model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ YOLOv8 model loading failed: {e}")
        print("  This is expected if the model is not downloaded yet")
        return False

def test_feature_extractor():
    """Test ResNet50 feature extractor"""
    try:
        from torchvision import models
        import torchvision.transforms as transforms

        # Create feature extractor
        feature_extractor = models.resnet50(pretrained=True)
        feature_extractor.eval()

        # Remove final classification layer
        feature_extractor = torch.nn.Sequential(
            *list(feature_extractor.children())[:-1]
        )

        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = feature_extractor(dummy_input)

        assert features.shape == (1, 2048, 1, 1)

        print("✓ ResNet50 feature extractor works correctly")
        return True
    except Exception as e:
        print(f"✗ Feature extractor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing PGO (Pose Graph Optimization) Module")
    print("=" * 50)

    tests = [
        ("PGO Module Import", test_pgo_import),
        ("YOLOv8 Import", test_yolo_import),
        ("g2o Import", test_g2o_import),
        ("Landmark Class", test_landmark_class),
        ("PoseGraph Classes", test_pose_graph_classes),
        ("Cosine Similarity", test_cosine_similarity),
        ("Relative Pose Computation", test_relative_pose_computation),
        ("YOLOv8 Model Loading", test_yolo_model_loading),
        ("Feature Extractor", test_feature_extractor),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! PGO module is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the installation.")
        print("\nNext steps:")
        print("1. Install missing dependencies")
        print("2. Run: ./install_pgo.sh")
        print("3. Re-run this test script")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
