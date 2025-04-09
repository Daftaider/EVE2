"""
UI components for the LCD display.
"""
import pygame
import logging
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class UIComponent:
    """Base class for UI components."""
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font):
        self.screen = screen
        self.font = font
        self.rect = pygame.Rect(0, 0, 0, 0)
        
    def draw(self) -> None:
        """Draw the component on the screen."""
        raise NotImplementedError
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle a pygame event."""
        return False

class VideoPanel(UIComponent):
    """Component for displaying video feed and object detection."""
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font, width: int, height: int):
        super().__init__(screen, font)
        self.width = width
        self.height = height
        self.frame = None
        self.detections = []
        self.rotation = 0
        
    def update_frame(self, frame: np.ndarray) -> None:
        """Update the video frame."""
        if frame is None:
            return
            
        # Apply rotation if needed
        if self.rotation != 0:
            if self.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
        # Calculate display area
        video_width = int(self.width * 0.8)
        video_height = int(video_width * frame.shape[0] / frame.shape[1])
        video_x = (self.width - video_width) // 2
        video_y = (self.height - video_height) // 2
        
        # Resize and convert frame
        frame = cv2.resize(frame, (video_width, video_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        
        self.frame = pygame.surfarray.make_surface(frame)
        self.rect = pygame.Rect(video_x, video_y, video_width, video_height)
        
    def update_detections(self, detections: List[Dict]) -> None:
        """Update object detections."""
        self.detections = detections
        
    def draw(self) -> None:
        """Draw the video panel with detections."""
        if self.frame is None:
            return
            
        # Draw frame
        self.screen.blit(self.frame, self.rect)
        
        # Draw detections
        for detection in self.detections:
            self._draw_detection(detection)
            
    def _draw_detection(self, detection: Dict) -> None:
        """Draw a single detection box and label."""
        try:
            # Get box coordinates
            if 'box' in detection:
                x1, y1, x2, y2 = detection['box']
            elif 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
            else:
                logger.warning(f"Invalid detection format: {detection}")
                return
                
            # Scale coordinates
            x1 = int(x1 * self.rect.width / self.frame.get_width()) + self.rect.x
            y1 = int(y1 * self.rect.height / self.frame.get_height()) + self.rect.y
            x2 = int(x2 * self.rect.width / self.frame.get_width()) + self.rect.x
            y2 = int(y2 * self.rect.height / self.frame.get_height()) + self.rect.y
            
            # Get label and confidence
            confidence = detection.get('confidence', 0.0)
            label = detection.get('label', 'unknown')
            if 'class' in detection:
                label = detection['class']
                
            # Draw box
            pygame.draw.rect(self.screen, (0, 255, 0), (x1, y1, x2-x1, y2-y1), 3)
            
            # Draw label with background
            text = f"{label}: {confidence:.2f}"
            if 'name' in detection:
                text = f"{detection['name']} ({text})"
                
            text_surface = self.font.render(text, True, (0, 255, 0))
            text_rect = text_surface.get_rect()
            
            bg_rect = pygame.Rect(x1, y1 - 25, text_rect.width + 10, text_rect.height + 10)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
            self.screen.blit(text_surface, (x1 + 5, y1 - 20))
            
            # Draw train button for person detections
            if label.lower() == 'person' and not detection.get('name'):
                train_button = pygame.Rect(x2 + 5, y1, 80, 30)
                pygame.draw.rect(self.screen, (255, 165, 0), train_button)
                train_text = self.font.render("Train", True, (0, 0, 0))
                self.screen.blit(train_text, (train_button.x + 10, train_button.y + 5))
                detection['train_button'] = train_button
                
        except Exception as e:
            logger.error(f"Error drawing detection: {e}", exc_info=True)

class ControlPanel(UIComponent):
    """Component for camera controls."""
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font, video_panel: VideoPanel):
        super().__init__(screen, font)
        self.video_panel = video_panel
        self.controls = {}
        
    def draw(self) -> None:
        """Draw the control panel."""
        if not self.video_panel.rect:
            return
            
        # Calculate panel position
        panel_width = self.screen.get_width() - self.video_panel.rect.width - self.video_panel.rect.x
        panel_x = self.video_panel.rect.x + self.video_panel.rect.width + 10
        panel_y = self.video_panel.rect.y
        panel_height = self.video_panel.rect.height
        
        # Draw panel background
        pygame.draw.rect(self.screen, (40, 40, 40), (panel_x, panel_y, panel_width, panel_height))
        
        # Draw controls
        control_y = panel_y + 20
        control_spacing = 50
        
        # Title
        title = self.font.render("Camera Controls", True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 10, control_y))
        control_y += control_spacing
        
        # Current rotation
        rotation_text = self.font.render(f"Rotation: {self.video_panel.rotation}°", True, (255, 255, 255))
        self.screen.blit(rotation_text, (panel_x + 10, control_y))
        control_y += control_spacing
        
        # Rotation buttons
        left_button = pygame.Rect(panel_x + 10, control_y, 40, 40)
        right_button = pygame.Rect(panel_x + 60, control_y, 40, 40)
        pygame.draw.rect(self.screen, (100, 100, 100), left_button)
        pygame.draw.rect(self.screen, (100, 100, 100), right_button)
        
        # Draw arrows
        left_arrow = self.font.render("←", True, (255, 255, 255))
        right_arrow = self.font.render("→", True, (255, 255, 255))
        self.screen.blit(left_arrow, (left_button.x + 10, left_button.y + 5))
        self.screen.blit(right_arrow, (right_button.x + 10, right_button.y + 5))
        control_y += control_spacing
        
        # Save button
        save_button = pygame.Rect(panel_x + 10, control_y, 90, 40)
        pygame.draw.rect(self.screen, (0, 200, 0), save_button)
        save_text = self.font.render("Save", True, (0, 0, 0))
        self.screen.blit(save_text, (save_button.x + 20, save_button.y + 10))
        
        # Store button rects
        self.controls = {
            'left_button': left_button,
            'right_button': right_button,
            'save_button': save_button
        }
        
        # Draw instructions
        instructions = [
            "Click on person to assign name",
            "Press 'Train' to learn face",
            "Use arrows to rotate",
            "Press 'Save' to keep rotation"
        ]
        
        control_y += control_spacing
        for instruction in instructions:
            text = self.font.render(instruction, True, (200, 200, 200))
            self.screen.blit(text, (panel_x + 10, control_y))
            control_y += 30
            
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle control panel events."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            if self.controls['left_button'].collidepoint(pos):
                self.video_panel.rotation = (self.video_panel.rotation - 90) % 360
                return True
            elif self.controls['right_button'].collidepoint(pos):
                self.video_panel.rotation = (self.video_panel.rotation + 90) % 360
                return True
            elif self.controls['save_button'].collidepoint(pos):
                # Save rotation logic here
                return True
        return False

class DebugOverlay(UIComponent):
    """Component for debug information overlay."""
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font):
        super().__init__(screen, font)
        self.fps = 0
        self.is_listening = False
        
    def update(self, fps: int, is_listening: bool) -> None:
        """Update debug information."""
        self.fps = fps
        self.is_listening = is_listening
        
    def draw(self) -> None:
        """Draw the debug overlay."""
        # Draw FPS counter
        fps_text = f"FPS: {self.fps}"
        fps_surface = self.font.render(fps_text, True, (255, 255, 255))
        self.screen.blit(fps_surface, (10, 10))
        
        # Draw listening status
        listening_text = "Listening: YES" if self.is_listening else "Listening: NO"
        listening_color = (0, 255, 0) if self.is_listening else (255, 0, 0)
        text_surface = self.font.render(listening_text, True, listening_color)
        self.screen.blit(text_surface, (10, 40))
        
        # Draw instructions
        instructions = [
            "ESC: Exit debug mode",
            "Double-click: Toggle debug menu",
            "Click on person: Assign name",
            "CTRL+S: Toggle debug mode",
            "←/→: Rotate camera",
            "S: Save rotation"
        ]
        
        y = 70
        for instruction in instructions:
            text_surface = self.font.render(instruction, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, y))
            y += 20 