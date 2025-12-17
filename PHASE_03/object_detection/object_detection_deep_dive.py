"""
Project 11: Object Detection Deep Dive
From image classification to real-world vision

This project demonstrates the evolution from basic image classification to sophisticated
object detection systems, implementing YOLO-style detection from scratch and using
state-of-the-art pre-trained models.

Learning Objectives:
1. Understand the difference between classification, localization, and detection
2. Implement basic object detection concepts
3. Work with bounding boxes and IoU calculations
4. Use pre-trained detection models (YOLO, SSD)
5. Create real-world detection applications

Business Context:
Object detection powers countless applications:
- Autonomous vehicles (detecting cars, pedestrians, signs)
- Retail (inventory management, checkout automation)
- Security (face detection, anomaly detection)
- Healthcare (medical image analysis)
- Manufacturing (quality control, defect detection)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# For advanced models - install with: pip install ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Note: Install ultralytics for advanced YOLO models: pip install ultralytics")

@dataclass
class BoundingBox:
    """Represents a bounding box with confidence and class"""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    class_name: str = ""
    
    @property
    def x_min(self) -> float:
        return self.x
    
    @property
    def y_min(self) -> float:
        return self.y
    
    @property
    def x_max(self) -> float:
        return self.x + self.width
    
    @property
    def y_max(self) -> float:
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width/2, self.y + self.height/2)
    
    @property
    def area(self) -> float:
        return self.width * self.height

class ObjectDetectionAnalyzer:
    """
    Comprehensive object detection analyzer demonstrating concepts from basic
    sliding windows to modern deep learning approaches
    """
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/detections", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
        # COCO class names for pre-trained models
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        IoU is the foundation metric for object detection:
        - Values range from 0 (no overlap) to 1 (perfect overlap)
        - Typically, IoU > 0.5 is considered a "good" detection
        """
        # Calculate intersection
        x_left = max(box1.x_min, box2.x_min)
        y_top = max(box1.y_min, box2.y_min)
        x_right = min(box1.x_max, box2.x_max)
        y_bottom = min(box1.y_max, box2.y_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = box1.area + box2.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def non_max_suppression(self, boxes: List[BoundingBox], 
                          iou_threshold: float = 0.5) -> List[BoundingBox]:
        """
        Apply Non-Maximum Suppression to eliminate redundant detections
        
        NMS is crucial for object detection because:
        - Multiple boxes often detect the same object
        - We want only the best detection per object
        - Reduces false positives and duplicate detections
        """
        if not boxes:
            return []
        
        # Sort boxes by confidence (highest first)
        boxes = sorted(boxes, key=lambda x: x.confidence, reverse=True)
        
        selected_boxes = []
        
        while boxes:
            # Take the box with highest confidence
            current_box = boxes.pop(0)
            selected_boxes.append(current_box)
            
            # Remove boxes with high IoU with current box
            boxes = [box for box in boxes 
                    if self.calculate_iou(current_box, box) < iou_threshold]
        
        return selected_boxes
    
    def create_synthetic_scene(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Create a synthetic scene with multiple objects for detection testing
        
        This simulates a real-world scenario where we need to detect
        multiple objects of different sizes and positions
        """
        # Create a complex background
        image = np.random.randint(20, 60, (height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for i in range(height):
            image[i, :, 0] += int(30 * np.sin(i * np.pi / height))
            image[i, :, 1] += int(20 * np.cos(i * np.pi / height))
        
        objects = []
        
        # Add different shaped objects
        for _ in range(np.random.randint(3, 7)):
            # Random position and size
            x = np.random.randint(50, width - 150)
            y = np.random.randint(50, height - 150)
            w = np.random.randint(40, 100)
            h = np.random.randint(40, 100)
            
            # Random color
            color = (np.random.randint(100, 255), 
                    np.random.randint(100, 255), 
                    np.random.randint(100, 255))
            
            # Draw rectangle or circle randomly
            if np.random.random() > 0.5:
                cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
                obj_type = "rectangle"
            else:
                center = (x + w//2, y + h//2)
                radius = min(w, h) // 2
                cv2.circle(image, center, radius, color, -1)
                obj_type = "circle"
            
            # Store object info for ground truth
            objects.append({
                'bbox': BoundingBox(x, y, w, h, 1.0, 0, obj_type),
                'type': obj_type,
                'color': color
            })
        
        return image, objects
    
    def sliding_window_detection(self, image: np.ndarray, 
                               window_sizes: List[Tuple[int, int]] = [(50, 50), (80, 80), (120, 120)],
                               step_size: int = 20) -> List[BoundingBox]:
        """
        Implement basic sliding window object detection
        
        This is the historical approach before deep learning:
        1. Slide windows of different sizes across the image
        2. Extract features from each window
        3. Classify each window as object/background
        
        While outdated, understanding this helps appreciate modern methods
        """
        detections = []
        height, width = image.shape[:2]
        
        for window_w, window_h in window_sizes:
            for y in range(0, height - window_h, step_size):
                for x in range(0, width - window_w, step_size):
                    # Extract window
                    window = image[y:y+window_h, x:x+window_w]
                    
                    # Simple feature: check if window has high variance (indicates object)
                    variance = np.var(window)
                    mean_intensity = np.mean(window)
                    
                    # Simple heuristic: objects have high variance and moderate intensity
                    confidence = 0.0
                    if variance > 500 and 50 < mean_intensity < 200:
                        # Calculate confidence based on variance and color distribution
                        color_std = np.std(window, axis=(0, 1))
                        confidence = min(0.9, variance / 2000.0 + np.mean(color_std) / 100.0)
                    
                    if confidence > 0.3:  # Threshold for detection
                        detections.append(BoundingBox(
                            x, y, window_w, window_h, confidence, 0, "object"
                        ))
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, 
                           detections: List[BoundingBox], 
                           title: str = "Detections",
                           ground_truth: List[BoundingBox] = None) -> None:
        """Visualize detection results with bounding boxes"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"{title} ({len(detections)} detections)", fontsize=14, fontweight='bold')
        
        # Draw ground truth in green
        if ground_truth:
            for gt_box in ground_truth:
                rect = patches.Rectangle(
                    (gt_box.x_min, gt_box.y_min), gt_box.width, gt_box.height,
                    linewidth=3, edgecolor='green', facecolor='none', linestyle='--'
                )
                ax.add_patch(rect)
                ax.text(gt_box.x_min, gt_box.y_min - 5, 
                       f"GT: {gt_box.class_name}", 
                       fontsize=8, color='green', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7))
        
        # Draw detections in red/blue based on confidence
        for detection in detections:
            # Color based on confidence
            color = 'red' if detection.confidence > 0.7 else 'blue'
            alpha = min(0.8, detection.confidence + 0.2)
            
            rect = patches.Rectangle(
                (detection.x_min, detection.y_min), detection.width, detection.height,
                linewidth=2, edgecolor=color, facecolor='none', alpha=alpha
            )
            ax.add_patch(rect)
            
            # Add confidence label
            ax.text(detection.x_min, detection.y_max + 15, 
                   f"{detection.class_name}: {detection.confidence:.2f}", 
                   fontsize=8, color=color, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/{title.lower().replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_basic_detection(self):
        """Demonstrate basic object detection concepts"""
        print("🎯 BASIC OBJECT DETECTION CONCEPTS")
        print("=" * 50)
        
        # Create synthetic scene
        print("\n1. Creating synthetic scene with multiple objects...")
        image, objects = self.create_synthetic_scene()
        ground_truth = [obj['bbox'] for obj in objects]
        
        # Apply sliding window detection
        print("2. Applying sliding window detection...")
        raw_detections = self.sliding_window_detection(image)
        print(f"   Raw detections: {len(raw_detections)}")
        
        # Apply Non-Maximum Suppression
        print("3. Applying Non-Maximum Suppression...")
        final_detections = self.non_max_suppression(raw_detections, iou_threshold=0.3)
        print(f"   Final detections after NMS: {len(final_detections)}")
        
        # Visualize results
        print("4. Visualizing results...")
        self.visualize_detections(image, final_detections, 
                                "Sliding Window Detection", ground_truth)
        
        # Calculate metrics
        print("\n5. Detection Metrics:")
        total_iou = 0
        matches = 0
        
        for detection in final_detections:
            best_iou = 0
            for gt_box in ground_truth:
                iou = self.calculate_iou(detection, gt_box)
                best_iou = max(best_iou, iou)
            
            if best_iou > 0.3:
                matches += 1
                total_iou += best_iou
        
        precision = matches / len(final_detections) if final_detections else 0
        recall = matches / len(ground_truth) if ground_truth else 0
        avg_iou = total_iou / matches if matches > 0 else 0
        
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   Average IoU: {avg_iou:.3f}")
        print(f"   F1-Score: {2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0:.3f}")
    
    def webcam_detection_demo(self, model_type: str = "haar"):
        """
        Real-time detection demo using webcam
        
        This demonstrates practical object detection applications:
        - Face detection using Haar cascades (classical CV)
        - Real-time processing considerations
        - User interaction and feedback
        """
        print(f"\n🎥 REAL-TIME DETECTION DEMO ({model_type.upper()})")
        print("=" * 50)
        print("Press 'q' to quit, 'c' to capture frame, 's' to switch model")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera. Using sample image instead.")
            # Create a sample face-like pattern
            sample_image = np.ones((480, 640, 3), dtype=np.uint8) * 50
            cv2.circle(sample_image, (320, 240), 80, (200, 180, 150), -1)  # Face
            cv2.circle(sample_image, (300, 220), 8, (0, 0, 0), -1)  # Left eye
            cv2.circle(sample_image, (340, 220), 8, (0, 0, 0), -1)  # Right eye
            cv2.ellipse(sample_image, (320, 260), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
            return sample_image
        
        # Load face detection cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        frame_count = 0
        fps_start = cv2.getTickCount()
        
        print("Camera initialized. Starting detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 0, 0), 2)
                
                # Detect eyes within face region
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    cv2.putText(roi_color, 'Eye', (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 1)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                fps_end = cv2.getTickCount()
                fps = 30.0 / ((fps_end - fps_start) / cv2.getTickFrequency())
                fps_start = fps_end
                print(f"FPS: {fps:.1f}, Faces detected: {len(faces)}")
            
            # Add overlay information
            cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, 'Press q to quit', (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Object Detection Demo', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                cv2.imwrite(f"{self.output_dir}/detections/captured_frame_{frame_count}.jpg", frame)
                print(f"Frame captured: captured_frame_{frame_count}.jpg")
            elif key == ord('s'):
                print("Model switching not implemented in this demo")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Detection demo completed!")
    
    def yolo_detection_demo(self):
        """
        Advanced YOLO detection demonstration
        
        YOLO (You Only Look Once) revolutionized object detection:
        - Single neural network predicts bounding boxes and class probabilities
        - Real-time performance with high accuracy
        - End-to-end trainable system
        """
        if not YOLO_AVAILABLE:
            print("❌ YOLO not available. Install with: pip install ultralytics")
            return
        
        print("\n🚀 ADVANCED YOLO DETECTION")
        print("=" * 50)
        
        try:
            # Load pre-trained YOLOv8 model
            print("Loading YOLOv8 model...")
            model = YOLO('yolov8n.pt')  # Nano version for speed
            
            # Create test image
            print("Creating test scene...")
            test_image, objects = self.create_synthetic_scene(800, 600)
            
            # Save test image
            cv2.imwrite(f"{self.output_dir}/test_image.jpg", test_image)
            
            # Run detection
            print("Running YOLO detection...")
            results = model(test_image)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.coco_classes[class_id] if class_id < len(self.coco_classes) else f"class_{class_id}"
                        
                        detection = BoundingBox(
                            x1, y1, x2-x1, y2-y1, confidence, class_id, class_name
                        )
                        detections.append(detection)
            
            # Visualize results
            self.visualize_detections(test_image, detections, "YOLO Detection Results")
            
            # Print detection summary
            print(f"\nDetection Summary:")
            print(f"Total detections: {len(detections)}")
            for detection in detections:
                print(f"  {detection.class_name}: {detection.confidence:.3f}")
            
        except Exception as e:
            print(f"❌ YOLO detection failed: {e}")
            print("This might happen if the model needs to be downloaded first.")
    
    def detection_metrics_analysis(self):
        """
        Comprehensive analysis of detection metrics and their business implications
        """
        print("\n📊 DETECTION METRICS DEEP DIVE")
        print("=" * 50)
        
        # Create multiple test scenarios
        scenarios = [
            {"name": "High Precision Scenario", "threshold": 0.8, "description": "Few false positives"},
            {"name": "High Recall Scenario", "threshold": 0.3, "description": "Catch most objects"},
            {"name": "Balanced Scenario", "threshold": 0.5, "description": "Balance precision/recall"}
        ]
        
        results = []
        
        for scenario in scenarios:
            print(f"\n{scenario['name']} (threshold={scenario['threshold']}):")
            print(f"Description: {scenario['description']}")
            
            # Create test scene
            image, objects = self.create_synthetic_scene()
            ground_truth = [obj['bbox'] for obj in objects]
            
            # Get detections
            raw_detections = self.sliding_window_detection(image)
            
            # Filter by threshold
            filtered_detections = [d for d in raw_detections if d.confidence >= scenario['threshold']]
            final_detections = self.non_max_suppression(filtered_detections, iou_threshold=0.3)
            
            # Calculate metrics
            tp = 0  # True positives
            fp = 0  # False positives
            fn = len(ground_truth)  # False negatives (start with all ground truth)
            
            matched_gt = set()
            
            for detection in final_detections:
                best_iou = 0
                best_gt_idx = -1
                
                for idx, gt_box in enumerate(ground_truth):
                    if idx not in matched_gt:
                        iou = self.calculate_iou(detection, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx
                
                if best_iou > 0.5:  # IoU threshold for positive match
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            fn = len(ground_truth) - len(matched_gt)
            
            # Calculate final metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            scenario_result = {
                'name': scenario['name'],
                'threshold': scenario['threshold'],
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            results.append(scenario_result)
            
            print(f"  Precision: {precision:.3f} (TP={tp}, FP={fp})")
            print(f"  Recall: {recall:.3f} (TP={tp}, FN={fn})")
            print(f"  F1-Score: {f1:.3f}")
            
            # Business interpretation
            print(f"  💼 Business Impact:")
            if precision > 0.8:
                print(f"     ✅ High precision: Low false alarms, reliable alerts")
            elif precision < 0.5:
                print(f"     ⚠️  Low precision: Many false alarms, user fatigue")
            
            if recall > 0.8:
                print(f"     ✅ High recall: Won't miss important objects")
            elif recall < 0.5:
                print(f"     ⚠️  Low recall: Missing many objects, safety concerns")
        
        # Visualize metrics comparison
        self.plot_metrics_comparison(results)
    
    def plot_metrics_comparison(self, results: List[Dict]):
        """Plot comparison of different detection scenarios"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        scenarios = [r['name'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1s = [r['f1'] for r in results]
        thresholds = [r['threshold'] for r in results]
        
        # Precision comparison
        bars1 = ax1.bar(scenarios, precisions, color=['red', 'blue', 'green'], alpha=0.7)
        ax1.set_title('Precision Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precision')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(precisions):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Recall comparison
        bars2 = ax2.bar(scenarios, recalls, color=['red', 'blue', 'green'], alpha=0.7)
        ax2.set_title('Recall Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Recall')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(recalls):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1-Score comparison
        bars3 = ax3.bar(scenarios, f1s, color=['red', 'blue', 'green'], alpha=0.7)
        ax3.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_ylim(0, 1)
        for i, v in enumerate(f1s):
            ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Precision-Recall curve
        ax4.plot(recalls, precisions, 'bo-', linewidth=2, markersize=8)
        ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        
        # Add threshold labels
        for i, (r, p, t) in enumerate(zip(recalls, precisions, thresholds)):
            ax4.annotate(f'T={t}', (r, p), xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/detection_metrics_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def business_applications_overview(self):
        """Overview of real-world object detection applications"""
        print("\n🏢 BUSINESS APPLICATIONS OF OBJECT DETECTION")
        print("=" * 60)
        
        applications = [
            {
                "domain": "Autonomous Vehicles",
                "objects": ["cars", "pedestrians", "traffic signs", "lane markers"],
                "metrics": {"precision": 0.999, "recall": 0.995},
                "cost_of_error": "High - accidents, liability",
                "requirements": "Real-time (30+ FPS), high accuracy"
            },
            {
                "domain": "Retail & E-commerce",
                "objects": ["products", "barcodes", "customers", "shopping carts"],
                "metrics": {"precision": 0.95, "recall": 0.90},
                "cost_of_error": "Medium - inventory errors, customer experience",
                "requirements": "Cost-effective, scalable"
            },
            {
                "domain": "Security & Surveillance",
                "objects": ["people", "weapons", "vehicles", "anomalies"],
                "metrics": {"precision": 0.85, "recall": 0.95},
                "cost_of_error": "High - security breaches, false alarms",
                "requirements": "24/7 operation, low false positives"
            },
            {
                "domain": "Healthcare",
                "objects": ["tumors", "fractures", "organs", "medical devices"],
                "metrics": {"precision": 0.98, "recall": 0.96},
                "cost_of_error": "Critical - patient safety, misdiagnosis",
                "requirements": "FDA approval, explainable AI"
            },
            {
                "domain": "Manufacturing",
                "objects": ["defects", "parts", "assembly", "workers"],
                "metrics": {"precision": 0.92, "recall": 0.88},
                "cost_of_error": "Medium - quality issues, recalls",
                "requirements": "Integration with existing systems"
            }
        ]
        
        for app in applications:
            print(f"\n📂 {app['domain']}:")
            print(f"   Detected Objects: {', '.join(app['objects'])}")
            print(f"   Required Metrics: Precision {app['metrics']['precision']:.1%}, Recall {app['metrics']['recall']:.1%}")
            print(f"   Cost of Error: {app['cost_of_error']}")
            print(f"   Special Requirements: {app['requirements']}")
            
            # ROI calculation example
            if app['domain'] == "Retail & E-commerce":
                print(f"   💰 ROI Example:")
                print(f"      - Implementation cost: $100,000")
                print(f"      - Annual labor savings: $200,000")
                print(f"      - Inventory accuracy improvement: 15%")
                print(f"      - Customer satisfaction increase: 8%")
                print(f"      - Payback period: 6 months")
    
    def run_complete_analysis(self):
        """Run the complete object detection analysis"""
        print("🎯 OBJECT DETECTION: FROM IMAGE CLASSIFICATION TO REAL-WORLD VISION")
        print("=" * 80)
        print("This comprehensive analysis covers:")
        print("1. Basic detection concepts (sliding windows, IoU, NMS)")
        print("2. Real-time detection with classical methods")
        print("3. Modern deep learning approaches (YOLO)")
        print("4. Metrics analysis and business implications")
        print("5. Real-world applications overview")
        print("=" * 80)
        
        try:
            # 1. Basic detection concepts
            self.demonstrate_basic_detection()
            
            # 2. Metrics analysis
            self.detection_metrics_analysis()
            
            # 3. YOLO demonstration (if available)
            self.yolo_detection_demo()
            
            # 4. Business applications
            self.business_applications_overview()
            
            print(f"\n✅ ANALYSIS COMPLETE!")
            print(f"📁 Results saved to: {self.output_dir}/")
            print("\n🎓 KEY TAKEAWAYS:")
            print("1. Object detection evolved from sliding windows to end-to-end deep learning")
            print("2. IoU and NMS are fundamental concepts for all detection systems")
            print("3. Precision vs Recall trade-offs depend on business requirements")
            print("4. Real-time performance requires careful model selection and optimization")
            print("5. Different applications have vastly different accuracy and cost requirements")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ObjectDetectionAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("🚀 NEXT STEPS:")
    print("1. Try the webcam demo for real-time detection")
    print("2. Experiment with different IoU and confidence thresholds")
    print("3. Implement custom object detection for your specific use case")
    print("4. Explore advanced architectures like YOLO, SSD, or R-CNN")
    print("5. Consider deployment challenges: model compression, edge computing")
    print("="*80)