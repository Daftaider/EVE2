"""
Display rendering for the LCD display.
"""
import pygame
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
from .display_state import DisplayState, DisplayMode

logger = logging.getLogger(__name__)

@dataclass
class RenderConfig:
    """Configuration for rendering."""
    font_size: int = 24
    line_height: int = 30
    margin: int = 10
    padding: int = 5
    border_width: int = 2
    corner_radius: int = 5

class DisplayRenderer:
    """Handles rendering of display elements."""
    
    def __init__(self, screen: pygame.Surface, state: DisplayState):
        """Initialize the display renderer."""
        self.screen = screen
        self.state = state
        self.config = RenderConfig()
        self.font = pygame.font.Font(None, self.config.font_size)
        
    def clear_screen(self) -> None:
        """Clear the screen with the background color."""
        self.screen.fill(self.state.background_color)
        
    def draw_text(self, text: str, position: Tuple[int, int], 
                 color: Tuple[int, int, int] = (255, 255, 255),
                 align: str = "left") -> None:
        """Draw text on the screen."""
        text_surface = self.font.render(text, True, color)
        if align == "center":
            position = (position[0] - text_surface.get_width() // 2, position[1])
        elif align == "right":
            position = (position[0] - text_surface.get_width(), position[1])
        self.screen.blit(text_surface, position)
        
    def draw_rect(self, rect: pygame.Rect, color: Tuple[int, int, int],
                 border_width: int = 0, corner_radius: int = 0) -> None:
        """Draw a rectangle on the screen."""
        pygame.draw.rect(self.screen, color, rect, border_width, corner_radius)
        
    def draw_circle(self, center: Tuple[int, int], radius: int,
                   color: Tuple[int, int, int], border_width: int = 0) -> None:
        """Draw a circle on the screen."""
        pygame.draw.circle(self.screen, color, center, radius, border_width)
        
    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int],
                 color: Tuple[int, int, int], width: int = 1) -> None:
        """Draw a line on the screen."""
        pygame.draw.line(self.screen, color, start, end, width)
        
    def draw_polygon(self, points: List[Tuple[int, int]],
                    color: Tuple[int, int, int], border_width: int = 0) -> None:
        """Draw a polygon on the screen."""
        pygame.draw.polygon(self.screen, color, points, border_width)
        
    def draw_arrow(self, start: Tuple[int, int], end: Tuple[int, int],
                  color: Tuple[int, int, int], width: int = 2,
                  arrow_size: int = 10) -> None:
        """Draw an arrow on the screen."""
        # Draw the main line
        self.draw_line(start, end, color, width)
        
        # Calculate arrow head points
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = pygame.math.Vector2(dx, dy).angle_to(pygame.math.Vector2(1, 0))
        
        # Calculate arrow head points
        arrow_angle = 30  # degrees
        arrow_points = [
            end,
            (end[0] - arrow_size * pygame.math.Vector2(1, 0).rotate(angle + arrow_angle).x,
             end[1] - arrow_size * pygame.math.Vector2(1, 0).rotate(angle + arrow_angle).y),
            (end[0] - arrow_size * pygame.math.Vector2(1, 0).rotate(angle - arrow_angle).x,
             end[1] - arrow_size * pygame.math.Vector2(1, 0).rotate(angle - arrow_angle).y)
        ]
        
        # Draw arrow head
        self.draw_polygon(arrow_points, color)
        
    def draw_fps(self, position: Tuple[int, int]) -> None:
        """Draw the FPS counter."""
        fps_text = f"FPS: {self.state.fps:.1f}"
        self.draw_text(fps_text, position)
        
    def draw_listening_indicator(self, position: Tuple[int, int]) -> None:
        """Draw the listening indicator."""
        if self.state.listening:
            self.draw_circle(position, 5, (0, 255, 0))
        else:
            self.draw_circle(position, 5, (255, 0, 0))
            
    def draw_debug_info(self, position: Tuple[int, int]) -> None:
        """Draw debug information."""
        if self.state.debug_info is None:
            return
            
        y = position[1]
        for key, value in self.state.debug_info.items():
            text = f"{key}: {value}"
            self.draw_text(text, (position[0], y))
            y += self.config.line_height
            
    def update(self) -> None:
        """Update the display based on the current state."""
        self.clear_screen()
        
        if self.state.mode == DisplayMode.NORMAL:
            # Draw normal display elements
            pass
        elif self.state.mode == DisplayMode.DEBUG:
            # Draw debug information
            self.draw_fps((self.config.margin, self.config.margin))
            self.draw_listening_indicator((self.screen.get_width() - self.config.margin,
                                         self.config.margin))
            self.draw_debug_info((self.config.margin, self.config.margin + self.config.line_height))
        elif self.state.mode == DisplayMode.MENU:
            # Draw menu elements
            pass
            
        pygame.display.flip() 