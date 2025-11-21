"""
Control Bridge: Converts gestures to game control commands
"""
from gesture_recognition import GestureRecognition


class ControlBridge:
    def __init__(self):
        """Initialize control bridge"""
        self.gesture_recognizer = GestureRecognition()
        
        # Current game state
        self.current_direction = None
        self.boost_active = False
        self.pause_triggered = False
        
        # Snake movement rules
        self.last_move_direction = None
        
    def update(self, head_position, keypoints):
        """
        Update control bridge with new head data
        
        Args:
            head_position: Current head position
            keypoints: Head keypoints dictionary
        """
        # Update gesture recognizer
        self.gesture_recognizer.update(head_position, keypoints)
        
        # Get direction based on head movement
        direction = self.gesture_recognizer.get_direction(keypoints)
        
        # Check for gestures
        is_open_palm = self.gesture_recognizer.is_open_palm(keypoints)
        is_closed_fist = self.gesture_recognizer.is_closed_fist(keypoints)
        
        # Update controls
        self._update_direction(direction)
        self._update_boost(is_open_palm)
        self._update_pause(is_closed_fist)
    
    def _update_direction(self, direction):
        """Update snake direction (prevent opposite moves)"""
        if direction is None:
            return
        
        # Prevent opposite direction moves (snake rule)
        opposite_map = {
            'UP': 'DOWN',
            'DOWN': 'UP',
            'LEFT': 'RIGHT',
            'RIGHT': 'LEFT'
        }
        
        if self.last_move_direction is None:
            # First move - allow any direction
            self.current_direction = direction
            self.last_move_direction = direction
        elif direction != opposite_map.get(self.last_move_direction):
            # Only update if not opposite direction
            self.current_direction = direction
            self.last_move_direction = direction
    
    def _update_boost(self, is_open_palm):
        """Update boost state"""
        if is_open_palm and self.gesture_recognizer.trigger_boost():
            self.boost_active = True
        else:
            self.boost_active = False
    
    def _update_pause(self, is_closed_fist):
        """Update pause state"""
        if is_closed_fist:
            if self.gesture_recognizer.trigger_pause():
                self.pause_triggered = True
        else:
            self.pause_triggered = False
    
    def get_direction(self):
        """Get current direction command"""
        return self.current_direction
    
    def get_boost(self):
        """Get boost state"""
        return self.boost_active
    
    def get_pause_trigger(self):
        """Get pause trigger (returns True once, then resets)"""
        if self.pause_triggered:
            self.pause_triggered = False
            return True
        return False
    
    def calibrate(self, head_position):
        """Calibrate neutral position"""
        self.gesture_recognizer.calibrate(head_position)
    
    def is_calibrated(self):
        """Check if calibrated"""
        return self.gesture_recognizer.is_calibrated
    
    def reset(self):
        """Reset control bridge"""
        self.gesture_recognizer.reset()
        self.current_direction = None
        self.boost_active = False
        self.pause_triggered = False
        self.last_move_direction = None

