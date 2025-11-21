"""
UI Module: Pygame rendering and display
"""
import pygame
import cv2
import numpy as np


class UIModule:
    def __init__(self, window_width=800, window_height=600):
        """
        Initialize UI module
        
        Args:
            window_width: Window width in pixels
            window_height: Window height in pixels
        """
        self.window_width = window_width
        self.window_height = window_height
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("AI-Controlled Snake Game - Head Movement")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.DARK_GREEN = (0, 200, 0)
        self.GRAY = (128, 128, 128)
        self.YELLOW = (255, 255, 0)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Camera preview settings (positioned in top-right corner to avoid game overlap)
        self.camera_preview_size = (240, 180)  # Smaller to fit better
        self.camera_preview_pos = (self.window_width - 250, 10)  # Top-right corner
        
    def draw_game(self, game, head_pos, keypoints, confidence, direction, boost_active, camera_frame=None):
        """
        Draw game screen
        
        Args:
            game: SnakeGame instance
            head_pos: Current head position
            keypoints: Head keypoints dictionary
            confidence: Detection confidence
            direction: Current direction
            boost_active: Whether boost is active
            camera_frame: Annotated camera frame from OpenCV
        """
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw game board
        self._draw_game_board(game)
        
        # Draw camera preview with live feed
        self._draw_camera_preview(head_pos, keypoints, confidence, camera_frame)
        
        # Draw UI elements
        self._draw_score(game.get_score())
        self._draw_direction_indicator(direction)
        self._draw_boost_indicator(boost_active)
        
        # Draw pause overlay
        if game.is_paused():
            self._draw_pause_overlay()
        
        # Draw game over overlay
        if game.is_game_over():
            self._draw_game_over_overlay(game.get_score())
        
        # Update display
        pygame.display.flip()
    
    def _draw_game_board(self, game):
        """Draw snake and food on game board"""
        grid_width, grid_height = game.get_grid_size()
        cell_size = game.get_cell_size()
        
        # Calculate board position (centered, accounting for camera preview)
        board_width = grid_width * cell_size
        board_height = grid_height * cell_size
        # Offset slightly left to avoid camera preview overlap
        board_x = (self.window_width - board_width) // 2 - 50
        board_y = (self.window_height - board_height) // 2 + 30
        
        # Draw board background
        pygame.draw.rect(self.screen, (20, 20, 20), 
                        (board_x - 5, board_y - 5, board_width + 10, board_height + 10))
        
        # Draw snake
        snake = game.get_snake()
        for i, segment in enumerate(snake):
            x = board_x + segment[0] * cell_size
            y = board_y + segment[1] * cell_size
            
            # Head is brighter
            color = self.GREEN if i == 0 else self.DARK_GREEN
            pygame.draw.rect(self.screen, color, (x, y, cell_size - 2, cell_size - 2))
        
        # Draw food
        food = game.get_food()
        food_x = board_x + food[0] * cell_size
        food_y = board_y + food[1] * cell_size
        pygame.draw.circle(self.screen, self.RED, 
                          (food_x + cell_size // 2, food_y + cell_size // 2),
                          cell_size // 2 - 2)
    
    def _draw_camera_preview(self, head_pos, keypoints, confidence, camera_frame=None):
        """Draw camera preview with live feed and head tracking"""
        preview_x, preview_y = self.camera_preview_pos
        preview_w, preview_h = self.camera_preview_size
        
        # Draw preview background
        pygame.draw.rect(self.screen, (30, 30, 30),
                        (preview_x, preview_y, preview_w, preview_h))
        
        # Display live camera feed if available
        if camera_frame is not None:
            try:
                # Resize frame to fit preview
                frame_resized = cv2.resize(camera_frame, (preview_w, preview_h))
                # Convert BGR to RGB for Pygame
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                # Convert to Pygame surface
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                self.screen.blit(frame_surface, (preview_x, preview_y))
            except Exception as e:
                # Fallback if frame conversion fails
                error_text = self.font_small.render("Camera Error", True, self.RED)
                self.screen.blit(error_text, (preview_x + 5, preview_y + 5))
        
        # Draw confidence indicator overlay
        if confidence > 0:
            conf_text = self.font_small.render(f"Conf: {confidence:.2f}", True, self.WHITE)
            # Draw semi-transparent background for text
            text_bg = pygame.Surface((conf_text.get_width() + 10, conf_text.get_height() + 4))
            text_bg.set_alpha(180)
            text_bg.fill(self.BLACK)
            self.screen.blit(text_bg, (preview_x + 5, preview_y + preview_h - 25))
            self.screen.blit(conf_text, (preview_x + 7, preview_y + preview_h - 23))
        
        # Draw detection status
        if head_pos is not None:
            status_text = self.font_small.render("HEAD DETECTED", True, self.GREEN)
        else:
            status_text = self.font_small.render("NO HEAD", True, self.RED)
        
        status_bg = pygame.Surface((status_text.get_width() + 10, status_text.get_height() + 4))
        status_bg.set_alpha(180)
        status_bg.fill(self.BLACK)
        self.screen.blit(status_bg, (preview_x + 5, preview_y + 5))
        self.screen.blit(status_text, (preview_x + 7, preview_y + 7))
    
    def _draw_score(self, score):
        """Draw score"""
        score_text = self.font_medium.render(f"Score: {score}", True, self.WHITE)
        self.screen.blit(score_text, (self.window_width - 150, 10))
    
    def _draw_direction_indicator(self, direction):
        """Draw current direction indicator"""
        if direction:
            dir_text = self.font_small.render(f"Direction: {direction}", True, self.YELLOW)
            self.screen.blit(dir_text, (self.window_width - 200, 50))
    
    def _draw_boost_indicator(self, boost_active):
        """Draw boost indicator"""
        if boost_active:
            boost_text = self.font_small.render("BOOST ACTIVE!", True, self.RED)
            self.screen.blit(boost_text, (self.window_width - 200, 80))
    
    def _draw_pause_overlay(self):
        """Draw pause overlay"""
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(128)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))
        
        pause_text = self.font_large.render("PAUSED", True, self.WHITE)
        text_rect = pause_text.get_rect(center=(self.window_width // 2, self.window_height // 2))
        self.screen.blit(pause_text, text_rect)
        
        hint_text = self.font_small.render("Close your fist to unpause", True, self.GRAY)
        hint_rect = hint_text.get_rect(center=(self.window_width // 2, self.window_height // 2 + 50))
        self.screen.blit(hint_text, hint_rect)
    
    def _draw_game_over_overlay(self, score):
        """Draw game over overlay"""
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(200)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.font_large.render("GAME OVER", True, self.RED)
        text_rect = game_over_text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 50))
        self.screen.blit(game_over_text, text_rect)
        
        score_text = self.font_medium.render(f"Final Score: {score}", True, self.WHITE)
        score_rect = score_text.get_rect(center=(self.window_width // 2, self.window_height // 2))
        self.screen.blit(score_text, score_rect)
        
        restart_text = self.font_small.render("Press SPACE to restart", True, self.GRAY)
        restart_rect = restart_text.get_rect(center=(self.window_width // 2, self.window_height // 2 + 50))
        self.screen.blit(restart_text, restart_rect)
    
    def draw_calibration(self, frame, countdown):
        """
        Draw calibration screen
        
        Args:
            frame: Camera frame
            countdown: Countdown seconds remaining
        """
        self.screen.fill(self.BLACK)
        
        # Convert OpenCV frame to Pygame surface if available
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (640, 480))
            frame_surface = pygame.surfarray.make_surface(frame_small.swapaxes(0, 1))
            self.screen.blit(frame_surface, 
                           ((self.window_width - 640) // 2, (self.window_height - 480) // 2))
        
        # Draw calibration text
        calib_text = self.font_large.render("CALIBRATION", True, self.WHITE)
        text_rect = calib_text.get_rect(center=(self.window_width // 2, 50))
        self.screen.blit(calib_text, text_rect)
        
        instruction_text = self.font_medium.render(
            "Hold your hand still in front of the camera", True, self.YELLOW)
        inst_rect = instruction_text.get_rect(center=(self.window_width // 2, 100))
        self.screen.blit(instruction_text, inst_rect)
        
        countdown_text = self.font_large.render(str(countdown), True, self.GREEN)
        countdown_rect = countdown_text.get_rect(center=(self.window_width // 2, self.window_height // 2))
        self.screen.blit(countdown_text, countdown_rect)
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle Pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def quit(self):
        """Clean up Pygame"""
        pygame.quit()

