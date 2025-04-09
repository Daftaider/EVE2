"""
Debug mode handling for the LCD display.
"""
import pygame
import logging
from typing import Optional, Dict, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)

class DebugMode(Enum):
    """Enum for different debug modes."""
    NONE = auto()
    VIDEO = auto()
    AUDIO = auto()

class DebugModeHandler:
    """Handles debug mode selection and display."""
    
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font):
        """Initialize the debug mode handler."""
        self.screen = screen
        self.font = font
        self.current_mode = DebugMode.NONE
        self.menu_visible = False
        
    def show_menu(self) -> None:
        """Show the debug mode selection menu."""
        self.menu_visible = True
        self._draw_menu()
        
    def hide_menu(self) -> None:
        """Hide the debug mode selection menu."""
        self.menu_visible = False
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events for debug mode selection."""
        if not self.menu_visible:
            return False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                self.current_mode = DebugMode.VIDEO
                self.menu_visible = False
                return True
            elif event.key == pygame.K_2:
                self.current_mode = DebugMode.AUDIO
                self.menu_visible = False
                return True
            elif event.key == pygame.K_3 or event.key == pygame.K_ESCAPE:
                self.current_mode = DebugMode.NONE
                self.menu_visible = False
                return True
                
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if click is within any option area
            pos = event.pos
            if self.video_rect.collidepoint(pos):
                self.current_mode = DebugMode.VIDEO
                self.menu_visible = False
                return True
            elif self.audio_rect.collidepoint(pos):
                self.current_mode = DebugMode.AUDIO
                self.menu_visible = False
                return True
            elif self.exit_rect.collidepoint(pos):
                self.current_mode = DebugMode.NONE
                self.menu_visible = False
                return True
                
        return False
        
    def _draw_menu(self) -> None:
        """Draw the debug mode selection menu."""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw title
        title = self.font.render("Debug Mode Selection", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 3))
        self.screen.blit(title, title_rect)
        
        # Draw options
        video_option = self.font.render("1. Video Debug (Object Detection)", True, (255, 255, 255))
        audio_option = self.font.render("2. Audio Debug (Voice Detection)", True, (255, 255, 255))
        exit_option = self.font.render("3. Exit Menu", True, (255, 255, 255))
        
        self.video_rect = video_option.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        self.audio_rect = audio_option.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 + 40))
        self.exit_rect = exit_option.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 + 80))
        
        self.screen.blit(video_option, self.video_rect)
        self.screen.blit(audio_option, self.audio_rect)
        self.screen.blit(exit_option, self.exit_rect)
        
        # Update display
        pygame.display.flip()
        
    def get_current_mode(self) -> DebugMode:
        """Get the current debug mode."""
        return self.current_mode
        
    def set_mode(self, mode: DebugMode) -> None:
        """Set the current debug mode."""
        self.current_mode = mode 