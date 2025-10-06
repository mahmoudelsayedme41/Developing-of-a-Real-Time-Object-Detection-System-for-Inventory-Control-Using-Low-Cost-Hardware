#!/usr/bin/env python3
# optimized_pi5_csi_bag_counter_with_zoom_FINAL.py
# Raspberry Pi 5 CSI Camera Bag Counter - Final Version
# Features: Fixed color handling, robust shutdown, zoom controls, and performance optimizations.

import os
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from datetime import datetime
from collections import deque
import signal
import sys
import gc
import psutil
import logging
from queue import Queue, Empty
import atexit

# -----------------------------------------------------------------------------
# CRITICAL: Performance Environment Setup
# -----------------------------------------------------------------------------
os.environ["PICAMERA2_PREVIEW"] = "NULL"
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019'] = '0'

from picamera2 import Picamera2

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Excel support check
EXCEL_AVAILABLE = False
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    pass

# -----------------------------------------------------------------------------
# OPTIMIZED CONFIGURATION
# -----------------------------------------------------------------------------
class Config:
    # Model Settings
    MODEL_PATH = "/home/pi/bag_counter/yolo_model/best.pt"
    MODEL_DEVICE = "cpu"
    MODEL_IMGSZ = 256

    # Camera Settings
    FRAME_WIDTH = 512
    FRAME_HEIGHT = 512
    CAPTURE_FPS = 18
    PROCESS_FPS = 8
    BUFFER_COUNT = 2

    # Detection Settings
    CONFIDENCE_THRESHOLD = 0.7
    IOU_THRESHOLD = 0.5
    MAX_DETECTIONS = 30

    # Tracking Settings
    TRACK_DISTANCE_THRESHOLD = 60
    TRACK_MIN_HITS = 2
    TRACK_MAX_AGE = 10

    # System Optimization
    MEMORY_CHECK_INTERVAL = 100
    MAX_MEMORY_USAGE = 75
    SAVE_INTERVAL = 60
    OUTPUT_DIR = "/home/pi/bag_counter/bag_counts/"

    # Display Optimization
    DISPLAY_SCALE = 0.8

# Class names for detected bags
class_names = [
    "14% رواد بياض دواجن",
    "14% رواد تسمين مواشي",
    "16% رواد حلا ب مواشي", 
    "16% رواد بياض دواجن",
    "16% رواد تسمين مواشي",
    "19% رواد حلا ب عالي الإدار مواشي",
    "19% رواد سوبر دواجن",
    "20% رواد فطام بتلو مواشي",
    "21% رواد سوبر دواجن",
    "21% رواد بادي نامي محبب دواجن",
    "21% رواد بادي نامي مفتت دواجن",
    "23% رواد سوبر دواجن"
]

# -----------------------------------------------------------------------------
# CAMERA SETUP AND CONTROL FUNCTIONS
# -----------------------------------------------------------------------------
def setup_camera():
    """Initialize and configure the Raspberry Pi Camera Module 2."""
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (Config.FRAME_WIDTH, Config.FRAME_HEIGHT), "format": "RGB888"},
            buffer_count=Config.BUFFER_COUNT,
            queue=False
        )
        picam2.configure(config)

        # Set camera controls for optimal color and exposure
        picam2.set_controls({
            "FrameRate": Config.CAPTURE_FPS,
            "AeEnable": True,     # Auto Exposure
            "AwbEnable": True,    # Auto White Balance
            "AwbMode": 0,         # Auto mode
            "Brightness": 0.05,
            "Contrast": 1.1,
            "Saturation": 1.2,
            "ColourGains": (1.4, 1.6),  # (Red, Blue) gains
        })

        picam2.start()
        print("Waiting for camera to stabilize colors...")
        time.sleep(2.0)

        # Verify camera output
        test_frame = picam2.capture_array()
        if test_frame is not None and len(test_frame.shape) == 3 and test_frame.shape[2] == 3:
            print("Camera initialized successfully. RGB format detected.")
        else:
            print("WARNING: Camera output format may be incorrect.")

        return picam2

    except Exception as e:
        logger.error(f"Camera setup failed: {e}")
        return None


def adjust_camera_settings(picam2, lighting_mode="indoor"):
    """Dynamically adjust camera settings based on lighting conditions."""
    base_settings = {
        "AeEnable": True,
        "AwbEnable": True,
        "Brightness": 0.05,
        "Contrast": 1.1,
        "Saturation": 1.2,
    }

    lighting_profiles = {
        "indoor": {"AwbMode": 1, "ColourGains": (1.5, 1.6)},
        "outdoor": {"AwbMode": 2, "ColourGains": (1.3, 1.7)},
        "fluorescent": {"AwbMode": 3, "ColourGains": (1.4, 1.5)},
        "led": {"AwbMode": 4, "ColourGains": (1.6, 1.4)},
        "enhance_red": {"AwbMode": 1, "Saturation": 1.5, "ColourGains": (1.8, 1.2), "Brightness": 0.1},
    }

    if lighting_mode in lighting_profiles:
        base_settings.update(lighting_profiles[lighting_mode])

    try:
        picam2.set_controls(base_settings)
        time.sleep(0.5)
        print(f"Applied '{lighting_mode}' lighting profile.")
    except Exception as e:
        print(f"Failed to apply camera settings: {e}")


# -----------------------------------------------------------------------------
# OPTIMIZED TRACKER CLASS
# -----------------------------------------------------------------------------
class FastTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.unique_bags = {name: set() for name in class_names}
        self.track_to_class = {}
        self.last_cleanup = time.time()
        # Pre-allocate arrays for performance
        self._temp_centers = np.zeros((Config.MAX_DETECTIONS, 2), dtype=np.float32)
        self._distances = np.zeros((100, Config.MAX_DETECTIONS), dtype=np.float32)

    def _fast_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, x2, y2 = box1[:4]
        x3, y3, x4, y4 = box2[:4]
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, detections):
        """Update object tracks based on new detections."""
        current_time = time.time()

        # Periodic cleanup of old tracks
        if current_time - self.last_cleanup > 5.0:
            dead_tracks = [tid for tid, t in self.tracks.items() if t['age'] > Config.TRACK_MAX_AGE]
            for tid in dead_tracks:
                del self.tracks[tid]
            self.last_cleanup = current_time

        # Limit number of detections for performance
        if len(detections) > Config.MAX_DETECTIONS:
            detections = sorted(detections, key=lambda x: x[4], reverse=True)[:Config.MAX_DETECTIONS]

        num_dets = len(detections)
        for i, det in enumerate(detections):
            self._temp_centers[i] = [(det[0] + det[2]) * 0.5, (det[1] + det[3]) * 0.5]

        unmatched_dets = set(range(num_dets))
        matched_tracks = set()

        # Match existing tracks to detections
        for tid, track in self.tracks.items():
            if track['age'] > Config.TRACK_MAX_AGE:
                continue

            last_bbox = track['bbox']
            track_center = np.array([(last_bbox[0] + last_bbox[2]) * 0.5, (last_bbox[1] + last_bbox[3]) * 0.5])
            best_match, best_score = -1, 0.0

            for i in unmatched_dets:
                det_center = self._temp_centers[i]
                dist = np.linalg.norm(track_center - det_center)
                if dist < Config.TRACK_DISTANCE_THRESHOLD:
                    iou = self._fast_iou(last_bbox, detections[i])
                    score = (1.0 - dist / Config.TRACK_DISTANCE_THRESHOLD) * 0.6 + iou * 0.4
                    if score > best_score and score > 0.3:
                        best_score, best_match = score, i

            if best_match != -1:
                det = detections[best_match]
                track.update({
                    'bbox': [int(det[0]), int(det[1]), int(det[2]), int(det[3])],
                    'confidence': det[4],
                    'class_id': int(det[5]),
                    'age': 0,
                    'hits': track['hits'] + 1,
                    'last_seen': current_time
                })
                unmatched_dets.discard(best_match)
                matched_tracks.add(tid)
            else:
                track['age'] += 1

        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            det = detections[i]
            self.tracks[self.next_id] = {
                'bbox': [int(det[0]), int(det[1]), int(det[2]), int(det[3])],
                'confidence': det[4],
                'class_id': int(det[5]),
                'age': 0,
                'hits': 1,
                'created_time': current_time,
                'last_seen': current_time
            }
            self.next_id += 1

        # Generate confirmed tracks and update unique bag counts
        confirmed_tracks = []
        for tid, track in self.tracks.items():
            if track['hits'] >= Config.TRACK_MIN_HITS and track['age'] == 0:
                confirmed_tracks.append([*track['bbox'], tid, track['confidence'], track['class_id']])
                cls_id = track['class_id']
                if (0 <= cls_id < len(class_names) and tid not in self.track_to_class and track['hits'] == Config.TRACK_MIN_HITS):
                    self.track_to_class[tid] = cls_id
                    self.unique_bags[class_names[cls_id]].add(tid)

        return confirmed_tracks


# -----------------------------------------------------------------------------
# ASYNCHRONOUS DATA SAVER
# -----------------------------------------------------------------------------
class AsyncSaver:
    def __init__(self):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        self.save_queue = Queue(maxsize=10)
        self.stop_flag = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def queue_save(self, unique_bags, suffix=""):
        """Queue a save operation."""
        try:
            data = {
                'unique_bags': {k: v.copy() for k, v in unique_bags.items()},
                'suffix': suffix,
                'timestamp': datetime.now()
            }
            self.save_queue.put_nowait(data)
        except:
            pass  # Queue full, skip this save

    def _worker(self):
        """Background worker thread for saving data."""
        while not self.stop_flag.is_set():
            try:
                data = self.save_queue.get(timeout=1.0)
                self._save_csv(data)
            except Empty:
                continue
            except Exception:
                continue

    def _save_csv(self, data):
        """Save data to a CSV file."""
        try:
            ts = data['timestamp'].strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(Config.OUTPUT_DIR, f"bag_count_{ts}{data['suffix']}.csv")
            total = sum(len(ids) for ids in data['unique_bags'].values() if ids)

            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Type,Count,Bag_IDs,Save_Time\n")
                for name, ids in data['unique_bags'].items():
                    if ids:
                        f.write(f'"{name}",{len(ids)},"{" ".join(map(str, sorted(ids)))}",{data["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'"TOTAL",{total},"",{data["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}\n')
        except Exception:
            pass

    def stop(self):
        """Stop the saver thread."""
        self.stop_flag.set()
        self.worker_thread.join(timeout=2)


# -----------------------------------------------------------------------------
# MAIN BAG COUNTER APPLICATION
# -----------------------------------------------------------------------------
class OptimizedBagCounter:
    def __init__(self):
        self.running = True
        self.model = None
        self.picam2 = None
        self.tracker = FastTracker()
        self.saver = AsyncSaver()

        # Performance counters
        self.frame_count = 0
        self.process_count = 0
        self.last_process_time = 0
        self.process_times = deque(maxlen=30)
        self.last_save_time = time.time()

        # Camera and Zoom State
        self.current_brightness = 0.05
        self.current_saturation = 1.2
        self.current_zoom = 1.0
        self.max_zoom = 4.0
        self.min_zoom = 1.0
        self.zoom_step = 0.2
        self.hardware_zoom_active = False
        self.mouse_zoom_mode = False

        # Register system signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False

    def _apply_digital_zoom(self, frame, zoom_factor):
        """Apply digital zoom to a frame."""
        if zoom_factor <= 1.0:
            return frame
        height, width = frame.shape[:2]
        crop_width = int(width / zoom_factor)
        crop_height = int(height / zoom_factor)
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2
        cropped = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
        return cv2.resize(cropped, (width, height))

    def _apply_hardware_zoom(self, zoom_factor):
        """Apply hardware zoom using the camera's scaler crop."""
        if not self.picam2:
            return False
        try:
            sensor_modes = self.picam2.sensor_modes
            full_width = sensor_modes[0]['size'][0] if sensor_modes else 4056
            full_height = sensor_modes[0]['size'][1] if sensor_modes else 3040

            if zoom_factor <= 1.0:
                self.picam2.set_controls({"ScalerCrop": None})
                self.hardware_zoom_active = False
                return True

            crop_width = int(full_width / zoom_factor)
            crop_height = int(full_height / zoom_factor)
            start_x = (full_width - crop_width) // 2
            start_y = (full_height - crop_height) // 2

            self.picam2.set_controls({"ScalerCrop": (start_x, start_y, crop_width, crop_height)})
            self.hardware_zoom_active = True
            print(f"Hardware zoom: {zoom_factor:.1f}x")
            return True

        except Exception as e:
            print(f"Hardware zoom failed: {e}. Using digital zoom.")
            self.hardware_zoom_active = False
            return False

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events for zooming to a region of interest."""
        if event == cv2.EVENT_LBUTTONDOWN and self.mouse_zoom_mode:
            if Config.DISPLAY_SCALE != 1.0:
                x = int(x / Config.DISPLAY_SCALE)
                y = int(y / Config.DISPLAY_SCALE)
            self._zoom_to_roi(x, y, Config.FRAME_WIDTH, Config.FRAME_HEIGHT, 2.0)
            self.mouse_zoom_mode = False
            print("Mouse zoom mode disabled.")

    def _zoom_to_roi(self, x, y, width, height, zoom_factor=2.0):
        """Zoom into a specific region of interest (ROI)."""
        try:
            sensor_modes = self.picam2.sensor_modes
            full_width = sensor_modes[0]['size'][0] if sensor_modes else 4056
            full_height = sensor_modes[0]['size'][1] if sensor_modes else 3040

            scale_x = full_width / width
            scale_y = full_height / height
            sensor_x = int(x * scale_x)
            sensor_y = int(y * scale_y)

            crop_width = int(full_width / zoom_factor)
            crop_height = int(full_height / zoom_factor)
            start_x = max(0, min(sensor_x - crop_width//2, full_width - crop_width))
            start_y = max(0, min(sensor_y - crop_height//2, full_height - crop_height))

            self.picam2.set_controls({"ScalerCrop": (start_x, start_y, crop_width, crop_height)})
            self.current_zoom = zoom_factor
            self.hardware_zoom_active = True
            print(f"Zoomed to ROI: {zoom_factor:.1f}x at ({x}, {y})")
        except Exception as e:
            print(f"ROI zoom failed: {e}")

    def _optimize_memory(self):
        """Perform garbage collection and cleanup old tracks if memory usage is high."""
        if self.frame_count % Config.MEMORY_CHECK_INTERVAL == 0:
            gc.collect()
            mem_percent = psutil.virtual_memory().percent
            if mem_percent > Config.MAX_MEMORY_USAGE:
                current_time = time.time()
                old_tracks = [tid for tid, t in self.tracker.tracks.items() if current_time - t.get('last_seen', 0) > 10]
                for tid in old_tracks:
                    del self.tracker.tracks[tid]
                gc.collect()
                return False
        return True

    def _show_help(self):
        """Display the user control help menu."""
        help_text = """
=== CAMERA CONTROLS ===
1 = Indoor lighting    2 = Outdoor lighting    3 = Fluorescent
4 = LED lighting       5 = Enhance red colors  r = Reset to auto
+/- = Brightness       c/v/b = Color saturation
w = Cycle white balance

=== ZOOM CONTROLS ===
z/x = Zoom in/out (0.2x)   Z/X = Fast zoom (0.5x)
0 = Reset zoom (1.0x)      9 = Max zoom (4.0x)
m = Mouse click zoom

=== OTHER CONTROLS ===
h = Show this help    s = Manual save
p = Show performance  q/ESC = Quit
========================
        """
        print(help_text)

    def initialize(self):
        """Initialize the YOLO model, camera, and display window."""
        if not os.path.exists(Config.MODEL_PATH):
            logger.error(f"Model not found: {Config.MODEL_PATH}")
            return False

        try:
            self.model = YOLO(Config.MODEL_PATH)
            # Warm up the model
            warmup = np.zeros((Config.MODEL_IMGSZ, Config.MODEL_IMGSZ, 3), dtype=np.uint8)
            _ = self.model(warmup, imgsz=Config.MODEL_IMGSZ, conf=Config.CONFIDENCE_THRESHOLD,
                          iou=Config.IOU_THRESHOLD, verbose=False, half=False)
            print("YOLO model loaded and warmed up.")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False

        self.picam2 = setup_camera()
        if not self.picam2:
            return False

        cv2.namedWindow("Pi5 Bag Counter with Zoom", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Pi5 Bag Counter with Zoom", self._mouse_callback)
        self._show_help()
        return True

    def _process_detections(self, frame):
        """
        Process a frame for object detection.
        FIX: Handles Camera Module 2's BGR output by converting it to RGB first.
        """
        try:
            # Camera Module 2 fix: Convert BGR to RGB, then to BGR for YOLO.
            frame_rgb_corrected = frame[:, :, ::-1]  # BGR -> RGB
            frame_bgr = cv2.cvtColor(frame_rgb_corrected, cv2.COLOR_RGB2BGR)  # RGB -> BGR for YOLO

            if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
                print(f"WARNING: Unexpected frame format: {frame_bgr.shape}")
                return [], frame_bgr

            results = self.model(
                frame_bgr,
                imgsz=Config.MODEL_IMGSZ,
                conf=Config.CONFIDENCE_THRESHOLD,
                iou=Config.IOU_THRESHOLD,
                verbose=False,
                half=False,
                device='cpu'
            )

            detections = []
            if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes.cpu().numpy()
                for box in boxes:
                    if len(box.xyxy) > 0:
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        detections.append([x1, y1, x2, y2, conf, cls_id])

            return detections, frame_bgr

        except Exception as e:
            logger.error(f"Detection error: {e}")
            # Fallback
            if len(frame.shape) == 3:
                return [], cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                return [], frame

    def _draw_optimized(self, frame, tracks):
        """Draw bounding boxes, IDs, and statistics on the frame."""
        height, width = frame.shape[:2]

        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2, tid, conf, cls_id = track
            color = (0, 255, 0) if conf > 0.8 else (0, 255, 255) if conf > 0.6 else (0, 150, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{tid}", (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw zoom indicator
        if self.current_zoom > 1.0:
            zoom_text = f"Zoom: {self.current_zoom:.1f}x ({'HW' if self.hardware_zoom_active else 'SW'})"
            cv2.putText(frame, zoom_text, (width - 180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            center_x, center_y = width // 2, height // 2
            cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 0), 2)
            cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 0), 2)

        # Draw performance stats
        total_bags = sum(len(ids) for ids in self.tracker.unique_bags.values())
        avg_time = np.mean(self.process_times) if self.process_times else 0
        fps = 1000 / avg_time if avg_time > 0 else 0

        stats_y = height - 100
        cv2.rectangle(frame, (5, stats_y), (320, height-5), (0, 0, 0), -1)
        stats = [
            f"Bags: {total_bags}",
            f"FPS: {fps:.1f} | {avg_time:.0f}ms",
            f"Zoom: {self.current_zoom:.1f}x",
            f"Sat: {self.current_saturation:.1f}"
        ]
        for i, text in enumerate(stats):
            cv2.putText(frame, text, (10, stats_y + 20 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _show_performance_stats(self):
        """Display detailed performance statistics in the console."""
        total_bags = sum(len(ids) for ids in self.tracker.unique_bags.values())
        avg_time = np.mean(self.process_times) if self.process_times else 0
        fps = 1000 / avg_time if avg_time > 0 else 0
        memory_usage = psutil.virtual_memory().percent

        stats = f"""
=== PERFORMANCE STATS ===
Frames Processed: {self.process_count}
Current FPS: {fps:.1f} | Avg Time: {avg_time:.1f}ms
Memory Usage: {memory_usage:.1f}%
Total Bags: {total_bags} | Active Tracks: {len(self.tracker.tracks)}
Zoom: {self.current_zoom:.1f}x ({'Hardware' if self.hardware_zoom_active else 'Software'})
=========================
        """
        print(stats)

    def run(self):
        """Main processing loop."""
        if not self.initialize():
            return False

        logger.warning("Starting optimized processing...")
        process_interval = 1.0 / Config.PROCESS_FPS

        try:
            while self.running:
                current_time = time.time()

                # Capture frame
                try:
                    frame = self.picam2.capture_array()
                    if frame is None or frame.size == 0 or len(frame.shape) != 3:
                        time.sleep(0.01)
                        continue
                except Exception as e:
                    print(f"Frame capture error: {e}")
                    time.sleep(0.1)
                    continue

                # Apply zoom if needed
                if self.current_zoom > 1.0 and not self.hardware_zoom_active:
                    frame = self._apply_digital_zoom(frame, self.current_zoom)

                self.frame_count += 1

                # Process frame for detections
                should_process = (current_time - self.last_process_time) >= process_interval
                if should_process:
                    start_time = time.time()
                    detections, frame_bgr = self._process_detections(frame)
                    tracks = self.tracker.update(detections)
                    self._draw_optimized(frame_bgr, tracks)
                    process_time = (time.time() - start_time) * 1000
                    self.process_times.append(process_time)
                    self.last_process_time = current_time
                    self.process_count += 1
                else:
                    # For display-only frames, apply the same color correction
                    frame_rgb_corrected = frame[:, :, ::-1]
                    frame_bgr = cv2.cvtColor(frame_rgb_corrected, cv2.COLOR_RGB2BGR)

                # Scale and display frame
                if Config.DISPLAY_SCALE != 1.0:
                    h, w = frame_bgr.shape[:2]
                    frame_bgr = cv2.resize(frame_bgr, (int(w * Config.DISPLAY_SCALE), int(h * Config.DISPLAY_SCALE)))

                cv2.imshow("Pi5 Bag Counter with Zoom", frame_bgr)
                key = cv2.waitKey(1) & 0xFF

                # Handle user input
                if key in [27, ord('q'), ord('Q')]:  # Exit
                    print("Exit command received.")
                    break
                elif key == ord('s'):  # Save
                    self.saver.queue_save(self.tracker.unique_bags, "_manual")
                    print("Manual save triggered.")
                elif key == ord('z'):  # Zoom in
                    new_zoom = min(self.current_zoom + self.zoom_step, self.max_zoom)
                    if new_zoom != self.current_zoom:
                        self.current_zoom = new_zoom
                        self._apply_hardware_zoom(self.current_zoom)
                elif key == ord('x'):  # Zoom out
                    new_zoom = max(self.current_zoom - self.zoom_step, self.min_zoom)
                    if new_zoom != self.current_zoom:
                        self.current_zoom = new_zoom
                        self._apply_hardware_zoom(self.current_zoom)
                elif key == ord('0'):  # Reset zoom
                    self.current_zoom = 1.0
                    self._apply_hardware_zoom(1.0)
                    print("Zoom reset to 1.0x.")
                elif key == ord('m'):  # Toggle mouse zoom
                    self.mouse_zoom_mode = not self.mouse_zoom_mode
                    status = "enabled" if self.mouse_zoom_mode else "disabled"
                    print(f"Mouse zoom mode {status}.")
                elif key == ord('1'):  # Lighting presets
                    adjust_camera_settings(self.picam2, "indoor")
                elif key == ord('5'):
                    adjust_camera_settings(self.picam2, "enhance_red")
                elif key == ord('h'):
                    self._show_help()
                elif key == ord('p'):
                    self._show_performance_stats()

                # Auto-save and memory optimization
                if current_time - self.last_save_time > Config.SAVE_INTERVAL:
                    self.saver.queue_save(self.tracker.unique_bags, "_auto")
                    self.last_save_time = current_time

                self._optimize_memory()

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        except Exception as e:
            print(f"Fatal error in run loop: {e}")
            return False
        finally:
            self.cleanup()

        return True

    def cleanup(self):
        """Clean up resources gracefully."""
        print("\nStarting cleanup process...")

        try:
            self.saver.queue_save(self.tracker.unique_bags, "_final")
            print("Final data save queued.")

            if self.picam2:
                try:
                    print("Resetting camera controls...")
                    # FIX: Robust ScalerCrop reset to prevent shutdown errors
                    try:
                        self.picam2.set_controls({"ScalerCrop": None})
                    except Exception as e:
                        print(f"ScalerCrop reset warning: {e}. Attempting fallback.")
                        try:
                            sensor_modes = self.picam2.sensor_modes
                            if sensor_modes:
                                w, h = sensor_modes[0]['size']
                                self.picam2.set_controls({"ScalerCrop": (0, 0, w, h)})
                        except Exception as fallback_e:
                            print(f"Fallback also failed: {fallback_e}. Proceeding with shutdown.")

                    self.picam2.stop()
                    self.picam2.close()
                    print("Camera closed successfully.")
                except Exception as e:
                    print(f"Camera cleanup error: {e}")

            cv2.destroyAllWindows()
            self.saver.stop()

            total_bags = sum(len(ids) for ids in self.tracker.unique_bags.values())
            print(f"\nFinal Stats: {self.process_count} frames processed, {total_bags} bags detected.")

        except Exception as e:
            print(f"Error during cleanup: {e}")

        print("Cleanup completed.")


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("   PI 5 CSI BAG COUNTER - FINAL VERSION")
    print("   Features: Color Fix, Zoom Controls, Robust Shutdown")
    print("=" * 70)

    counter = OptimizedBagCounter()
    exit_code = 0

    try:
        success = counter.run()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n=== PROGRAM INTERRUPTED BY USER ===")
        exit_code = 0
    except Exception as e:
        print(f"\n=== FATAL ERROR ===\n{e}")
        exit_code = 1
    finally:
        print(f"\n=== PROGRAM ENDED (Exit Code: {exit_code}) ===")
        sys.exit(exit_code)