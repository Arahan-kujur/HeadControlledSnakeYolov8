"""
Gesture Recognition Module: Analyzes hand movements and gestures
"""
import numpy as np
from collections import deque


class GestureRecognition:
    def __init__(self, movement_threshold=30, history_size=10):
        """
        Initialize gesture recognition
        
        Args:
            movement_threshold: Minimum pixel movement to register direction change
            history_size: Number of positions to track for movement analysis
        """
        self.movement_threshold = movement_threshold
        self.history_size = history_size
        
        # Position history for movement tracking
        self.position_history = deque(maxlen=history_size)
        
        # Calibration
        self.neutral_position = None
        self.is_calibrated = False
        
        # Gesture state
        self.current_direction = None
        self.last_direction = None
        
        # Gesture cooldowns
        self.pause_cooldown = 0
        self.boost_cooldown = 0
        
    def calibrate(self, head_position):
        """
        Set neutral position for calibration
        
        Args:
            head_position: (x, y) tuple of head position
        """
        if head_position is not None:
            self.neutral_position = head_position
            self.is_calibrated = True
            self.position_history.clear()
            self.position_history.append(head_position)
            print("Calibration complete!")
    
    def update(self, head_position, keypoints):
        """
        Update gesture recognition with new head data
        
        Args:
            head_position: Current head position (x, y) - nose position
            keypoints: Dictionary of head keypoints
        """
        if head_position is None:
            return
        
        # Add to history
        self.position_history.append(head_position)
        
        # Update cooldowns
        if self.pause_cooldown > 0:
            self.pause_cooldown -= 1
        if self.boost_cooldown > 0:
            self.boost_cooldown -= 1
    
    def get_direction(self, keypoints=None):
        """
        Determine movement direction based on head movement
        
        Args:
            keypoints: Dictionary of head keypoints (head/nose position)
        
        Returns:
            str: 'UP', 'DOWN', 'LEFT', 'RIGHT', or None
        """
        if len(self.position_history) < 3:
            return None
        
        # Compare recent head position to older position
        recent = np.array(self.position_history[-1])
        older = np.array(self.position_history[0])
        
        movement = recent - older
        
        # Calculate movement magnitude
        dx = movement[0]
        dy = movement[1]
        
        # Determine dominant direction
        if abs(dx) > abs(dy):
            # Horizontal movement
            if abs(dx) < self.movement_threshold:
                return None
            if dx > 0:
                direction = 'RIGHT'
            else:
                direction = 'LEFT'
        else:
            # Vertical movement
            if abs(dy) < self.movement_threshold:
                return None
            if dy > 0:
                direction = 'DOWN'
            else:
                direction = 'UP'
        
        self.last_direction = self.current_direction
        self.current_direction = direction
        
        return direction
    
    def is_open_palm(self, keypoints):
        """
        Detect if hand is open (palm spread)
        
        Args:
            keypoints: Dictionary of hand keypoints
            
        Returns:
            bool: True if open palm detected
        """
        if keypoints is None:
            return False
        
        # Check if we have enough keypoints
        finger_tips = ['thumb', 'index', 'middle', 'ring', 'pinky']
        valid_fingers = [kp for kp in finger_tips if keypoints.get(kp) is not None]
        
        if len(valid_fingers) < 3:
            return False
        
        # Calculate distances between fingertips
        wrist = keypoints.get('wrist')
        if wrist is None:
            return False
        
        finger_positions = []
        for finger in valid_fingers:
            if keypoints[finger] is not None:
                finger_positions.append(keypoints[finger])
        
        if len(finger_positions) < 3:
            return False
        
        # Calculate average distance from wrist to fingertips
        distances = []
        for finger_pos in finger_positions:
            dist = np.sqrt((finger_pos[0] - wrist[0])**2 + (finger_pos[1] - wrist[1])**2)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # Calculate spread between fingertips
        if len(finger_positions) >= 3:
            # Distance between index and pinky (hand width)
            index_pinky_dist = np.sqrt(
                (finger_positions[0][0] - finger_positions[-1][0])**2 +
                (finger_positions[0][1] - finger_positions[-1][1])**2
            )
            
            # Open palm: fingers spread wide and extended
            # Threshold: average finger distance > 40 pixels and spread > 50 pixels
            return avg_distance > 40 and index_pinky_dist > 50
        
        return False
    
    def is_closed_fist(self, keypoints):
        """
        Detect if hand is closed (fist)
        
        Args:
            keypoints: Dictionary of hand keypoints
            
        Returns:
            bool: True if closed fist detected
        """
        if keypoints is None:
            return False
        
        wrist = keypoints.get('wrist')
        if wrist is None:
            return False
        
        # Check finger positions relative to wrist
        finger_tips = ['thumb', 'index', 'middle', 'ring', 'pinky']
        valid_fingers = [kp for kp in finger_tips if keypoints.get(kp) is not None]
        
        if len(valid_fingers) < 3:
            return False
        
        # Calculate distances from wrist to fingertips
        distances = []
        for finger in valid_fingers:
            if keypoints[finger] is not None:
                dist = np.sqrt(
                    (keypoints[finger][0] - wrist[0])**2 +
                    (keypoints[finger][1] - wrist[1])**2
                )
                distances.append(dist)
        
        if len(distances) == 0:
            return False
        
        avg_distance = np.mean(distances)
        
        # Closed fist: fingertips close to wrist
        # Threshold: average distance < 30 pixels
        return avg_distance < 30
    
    def trigger_pause(self):
        """Trigger pause gesture (with cooldown)"""
        if self.pause_cooldown == 0:
            self.pause_cooldown = 30  # 1 second cooldown at 30 FPS
            return True
        return False
    
    def trigger_boost(self):
        """Trigger boost gesture (with cooldown)"""
        if self.boost_cooldown == 0:
            self.boost_cooldown = 10  # Short cooldown
            return True
        return False
    
    def reset(self):
        """Reset gesture recognition state"""
        self.position_history.clear()
        self.current_direction = None
        self.last_direction = None
        self.is_calibrated = False
        self.neutral_position = None

