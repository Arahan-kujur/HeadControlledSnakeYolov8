"""
Main Controller: Integrates all modules
"""
import time
import cv2
import pygame
from vision import VisionModule
from control_bridge import ControlBridge
from game_engine import SnakeGame
from ui import UIModule


class GameController:
    def __init__(self):
        """Initialize game controller"""
        self.vision = VisionModule(camera_id=0)
        self.control_bridge = ControlBridge()
        self.game = SnakeGame(grid_width=20, grid_height=20, cell_size=20)
        self.ui = UIModule(window_width=800, window_height=600)
        
        self.running = True
        self.calibrated = False
        self.calibration_countdown = 3
        
    def initialize(self):
        """Initialize all modules"""
        print("Initializing game...")
        try:
            self.vision.initialize()
            print("All modules initialized successfully!")
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
    
    def run_calibration(self):
        """Run calibration sequence"""
        print("Starting calibration...")
        calibration_start = time.time()
        calibration_duration = 3.0  # 3 seconds
        
        last_countdown = 3
        
        while time.time() - calibration_start < calibration_duration:
            # Get frame
            frame = self.vision.get_frame()
            if frame is None:
                continue
            
            # Detect hand (returns annotated frame)
            wrist_pos, keypoints, confidence, annotated_frame = self.vision.detect_hand(frame)
            
            # Update countdown display
            elapsed = time.time() - calibration_start
            remaining = calibration_duration - elapsed
            current_countdown = int(remaining) + 1
            
            if current_countdown != last_countdown:
                print(f"Calibrating... {current_countdown}")
                last_countdown = current_countdown
            
            # Draw calibration screen with annotated frame
            self.ui.draw_calibration(annotated_frame if annotated_frame is not None else frame, current_countdown)
            
            # Check for quit
            if not self.ui.handle_events():
                return False
        
        # Final calibration
        frame = self.vision.get_frame()
        wrist_pos, keypoints, confidence, annotated_frame = self.vision.detect_hand(frame)
        
        if wrist_pos is not None:
            self.control_bridge.calibrate(wrist_pos)
            self.calibrated = True
            print("Calibration complete!")
            time.sleep(1)  # Brief pause before game starts
            return True
        else:
            print("Warning: No hand detected during calibration")
            # Continue anyway
            self.calibrated = True
            return True
    
    def run_game_loop(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        FPS = 60
        
        print("Game started! Use head movement to control the snake.")
        print("Controls:")
        print("  - Move head up/down/left/right: Change direction")
        print("  - Open palm: Speed boost")
        print("  - Closed fist: Pause/Unpause")
        
        while self.running:
            # Handle events
            if not self.ui.handle_events():
                break
            
            # Get camera frame
            frame = self.vision.get_frame()
            if frame is None:
                continue
            
            # Detect head (returns annotated frame with boxes and keypoints)
            head_pos, keypoints, confidence, annotated_frame = self.vision.detect_hand(frame)
            
            # Update control bridge
            self.control_bridge.update(head_pos, keypoints)
            
            # Get control commands
            direction = self.control_bridge.get_direction()
            boost_active = self.control_bridge.get_boost()
            pause_trigger = self.control_bridge.get_pause_trigger()
            
            # Handle pause
            if pause_trigger:
                self.game.toggle_pause()
            
            # Update game
            if not self.game.is_paused():
                self.game.update(direction, boost_active)
            
            # Check for restart (space key or game over + gesture)
            if self.game.is_game_over():
                # Could add gesture-based restart here
                pass
            
            # Draw everything with live camera feed
            self.ui.draw_game(
                self.game,
                head_pos,
                keypoints,
                confidence,
                direction,
                boost_active,
                annotated_frame
            )
            
            # Handle keyboard restart
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] and self.game.is_game_over():
                self.game.reset()
                self.control_bridge.reset()
            
            # Maintain FPS
            clock.tick(FPS)
    
    def run(self):
        """Run the complete game"""
        if not self.initialize():
            return
        
        try:
            # Calibration
            if not self.run_calibration():
                return
            
            # Main game loop
            self.run_game_loop()
            
        except KeyboardInterrupt:
            print("\nGame interrupted by user")
        except Exception as e:
            print(f"Error during gameplay: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.vision.release()
        self.ui.quit()
        print("Cleanup complete")


if __name__ == "__main__":
    controller = GameController()
    controller.run()

