"""
Game Engine: Snake game logic
"""
import random
import numpy as np


class SnakeGame:
    def __init__(self, grid_width=20, grid_height=20, cell_size=20):
        """
        Initialize Snake game
        
        Args:
            grid_width: Number of cells horizontally
            grid_height: Number of cells vertically
            cell_size: Size of each cell in pixels
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        
        # Game state
        self.reset()
        
        # Game settings
        self.base_speed = 10  # Base frames per move
        self.boost_speed = 5  # Boost frames per move
        self.speed_counter = 0
        
    def reset(self):
        """Reset game to initial state"""
        # Snake starts in center
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        
        self.snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y)
        ]
        
        self.direction = 'RIGHT'
        self.next_direction = 'RIGHT'
        self.score = 0
        self.game_over = False
        self.paused = False
        
        # Spawn initial food
        self.food = self._spawn_food()
        
    def update(self, direction_command, boost_active):
        """
        Update game state
        
        Args:
            direction_command: Direction from control bridge ('UP', 'DOWN', 'LEFT', 'RIGHT')
            boost_active: Whether boost is active
        """
        if self.game_over or self.paused:
            return
        
        # Update direction if valid
        if direction_command is not None:
            opposite = {
                'UP': 'DOWN',
                'DOWN': 'UP',
                'LEFT': 'RIGHT',
                'RIGHT': 'LEFT'
            }
            # Prevent opposite direction
            if direction_command != opposite.get(self.direction):
                self.next_direction = direction_command
        
        # Determine speed
        current_speed = self.boost_speed if boost_active else self.base_speed
        
        # Move snake based on speed
        self.speed_counter += 1
        if self.speed_counter >= current_speed:
            self.speed_counter = 0
            self._move_snake()
    
    def _move_snake(self):
        """Move snake one step"""
        # Update direction
        self.direction = self.next_direction
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        
        direction_map = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0)
        }
        
        dx, dy = direction_map[self.direction]
        new_head = (head_x + dx, head_y + dy)
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or
            new_head[1] < 0 or new_head[1] >= self.grid_height):
            self.game_over = True
            return
        
        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            return
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check food collision
        if new_head == self.food:
            self.score += 10
            self.food = self._spawn_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
    
    def _spawn_food(self):
        """Spawn food at random location (not on snake)"""
        while True:
            food_x = random.randint(0, self.grid_width - 1)
            food_y = random.randint(0, self.grid_height - 1)
            food_pos = (food_x, food_y)
            
            if food_pos not in self.snake:
                return food_pos
    
    def toggle_pause(self):
        """Toggle pause state"""
        if not self.game_over:
            self.paused = not self.paused
    
    def get_snake(self):
        """Get snake body positions"""
        return self.snake
    
    def get_food(self):
        """Get food position"""
        return self.food
    
    def get_score(self):
        """Get current score"""
        return self.score
    
    def is_game_over(self):
        """Check if game is over"""
        return self.game_over
    
    def is_paused(self):
        """Check if game is paused"""
        return self.paused
    
    def get_grid_size(self):
        """Get grid dimensions"""
        return (self.grid_width, self.grid_height)
    
    def get_cell_size(self):
        """Get cell size in pixels"""
        return self.cell_size

