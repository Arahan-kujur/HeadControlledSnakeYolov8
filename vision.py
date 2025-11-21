"""
Vision Module: Handles webcam input and YOLOv8 pose detection for head tracking
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque


class VisionModule:
    def __init__(self, camera_id=0, model_size='s', smoothing_window=5):
        """
        Initialize vision module with YOLOv8 pose model for head tracking
        
        Args:
            camera_id: Webcam device ID (default 0)
            model_size: Model size ('n', 's', 'm', 'l', 'x') - 's' recommended
            smoothing_window: Number of frames for rolling average smoothing
        """
        self.camera_id = camera_id
        self.model_size = model_size
        self.cap = None
        self.model = None
        self.device = 'cpu'  # Will be set to 'cuda' if GPU available
        self.smoothing_window = smoothing_window
        
        # COCO pose keypoint indices for head
        # Nose = 0, Left eye = 1, Right eye = 2
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        
        # Smoothing buffers
        self.head_history = deque(maxlen=smoothing_window)
        self.keypoints_history = deque(maxlen=smoothing_window)
        
        # Current detection state
        self.current_head = None
        self.current_keypoints = None
        self.detection_confidence = 0.0
        
    def _find_available_camera(self, start_index=0, max_tries=5):
        """
        Auto-detect available camera by trying different indices
        
        Args:
            start_index: Starting camera index to try
            max_tries: Maximum number of camera indices to try
            
        Returns:
            int: Camera index that works, or None if none found
        """
        print("Searching for available camera...")
        
        # Try the specified camera_id first
        if start_index >= 0:
            cap = cv2.VideoCapture(start_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    print(f"Found camera at index {start_index}")
                    return start_index
                cap.release()
        
        # Try other indices
        for i in range(max_tries):
            if i == start_index:
                continue
            print(f"Trying camera index {i}...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    print(f"Found camera at index {i}")
                    return i
                cap.release()
        
        return None
    
    def initialize(self, auto_detect=True):
        """
        Initialize camera and YOLOv8 pose model
        
        Args:
            auto_detect: If True, automatically find available camera
        """
        # Load YOLOv8 pose model
        model_name = f'yolov8{self.model_size}-pose.pt'
        print(f"Loading YOLOv8 pose model: {model_name}")
        self.model = YOLO(model_name)
        
        # Set device to GPU if available
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("Using CPU (GPU not available)")
        
        # Auto-detect camera if enabled
        if auto_detect:
            found_camera_id = self._find_available_camera(self.camera_id)
            if found_camera_id is None:
                raise RuntimeError(
                    f"No working camera found. Tried indices 0-4.\n"
                    f"Please check:\n"
                    f"  - Camera is connected\n"
                    f"  - No other application is using the camera\n"
                    f"  - Camera drivers are installed"
                )
            self.camera_id = found_camera_id
        else:
            # Try to use specified camera
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cap.release()
                raise RuntimeError(
                    f"Camera {self.camera_id} opened but cannot read frames.\n"
                    f"Try enabling auto_detect=True or use a different camera index."
                )
            self.cap.release()
        
        # Initialize webcam with found index
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera initialized successfully at index {self.camera_id}")
        
    def get_frame(self):
        """Capture and return current frame (mirrored horizontally)"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Flip frame horizontally to fix mirroring
        frame = cv2.flip(frame, 1)
        return frame
    
    def detect_hand(self, frame):
        """
        Detect head position using YOLOv8 pose (kept name for compatibility)
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            tuple: (head_position, keypoints_dict, confidence, annotated_frame)
        """
        if frame is None:
            return None, None, 0.0, None
        
        # Run YOLOv8 pose detection with GPU acceleration
        results = self.model(frame, device=self.device, verbose=False)
        
        head_pos = None
        keypoints = {}
        confidence = 0.0
        
        if len(results) > 0 and results[0].keypoints is not None:
            # Check if keypoints data exists and has elements
            try:
                keypoints_data_array = results[0].keypoints.data
                if keypoints_data_array is None or len(keypoints_data_array) == 0:
                    # No detections
                    annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()
                    return None, None, 0.0, annotated_frame
                
                keypoints_data = keypoints_data_array[0]  # First person detected
                
                # Verify keypoints_data has valid shape
                if keypoints_data is None or keypoints_data.shape[0] <= self.NOSE:
                    # Create annotated frame even without detection
                    annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()
                    return None, None, 0.0, annotated_frame
            except (IndexError, AttributeError, TypeError) as e:
                # Handle any indexing errors gracefully
                annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()
                return None, None, 0.0, annotated_frame
            
            # Extract head keypoints (nose position)
            if keypoints_data.shape[0] > self.NOSE:
                # Get nose position (head center)
                nose = keypoints_data[self.NOSE].cpu().numpy()
                if nose[2] > 0.3:  # Confidence threshold
                    head_pos = (int(nose[0]), int(nose[1]))
                    confidence = nose[2]
                    
                    # Extract other head keypoints for visualization
                    keypoints['head'] = head_pos
                    keypoints['nose'] = head_pos
                    keypoints['left_eye'] = self._get_keypoint(keypoints_data, self.LEFT_EYE)
                    keypoints['right_eye'] = self._get_keypoint(keypoints_data, self.RIGHT_EYE)
        
        # Create annotated frame with bounding boxes and keypoints
        try:
            if len(results) > 0:
                # Use YOLOv8's built-in plotting for bounding boxes and skeleton
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame.copy()
        except Exception:
            annotated_frame = frame.copy()
        
        # Draw additional head visualization
        if head_pos is not None:
            # Draw head position circle
            cv2.circle(annotated_frame, head_pos, 15, (0, 255, 0), -1)
            cv2.circle(annotated_frame, head_pos, 20, (0, 255, 0), 2)
            
            # Add text overlay showing head detected
            cv2.putText(annotated_frame, "HEAD DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Move your head to control", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            # Show no head detected message
            cv2.putText(annotated_frame, "NO HEAD DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, "Face the camera", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Apply smoothing
        if head_pos is not None:
            self.head_history.append(head_pos)
            self.keypoints_history.append(keypoints)
            
            # Use smoothed position
            if len(self.head_history) > 0:
                smoothed_head = np.mean(self.head_history, axis=0)
                self.current_head = (int(smoothed_head[0]), int(smoothed_head[1]))
                self.current_keypoints = self._smooth_keypoints()
                self.detection_confidence = confidence
        else:
            # No detection - use last known position if available
            if len(self.head_history) > 0:
                self.current_head = tuple(self.head_history[-1])
                self.current_keypoints = self.keypoints_history[-1] if len(self.keypoints_history) > 0 else None
                self.detection_confidence = 0.0
        
        return self.current_head, self.current_keypoints, self.detection_confidence, annotated_frame
    
    def _get_keypoint(self, keypoints_data, index):
        """Extract keypoint if confidence is sufficient"""
        if keypoints_data.shape[0] > index:
            kp = keypoints_data[index].cpu().numpy()
            if kp[2] > 0.3:  # Confidence threshold
                return (int(kp[0]), int(kp[1]))
        return None
    
    def _smooth_keypoints(self):
        """Average keypoints over smoothing window"""
        if len(self.keypoints_history) == 0:
            return None
        
        # Average positions for each keypoint
        smoothed = {}
        for key in ['head', 'nose', 'left_eye', 'right_eye']:
            positions = []
            for kp_dict in self.keypoints_history:
                if kp_dict and key in kp_dict and kp_dict[key] is not None:
                    positions.append(kp_dict[key])
            
            if len(positions) > 0:
                avg_pos = np.mean(positions, axis=0)
                smoothed[key] = (int(avg_pos[0]), int(avg_pos[1]))
            else:
                smoothed[key] = None
        
        return smoothed
    
    def get_wrist_position(self):
        """Get current smoothed head position (kept name for compatibility)"""
        return self.current_head
    
    def get_keypoints(self):
        """Get current smoothed keypoints"""
        return self.current_keypoints
    
    def get_confidence(self):
        """Get detection confidence"""
        return self.detection_confidence
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
