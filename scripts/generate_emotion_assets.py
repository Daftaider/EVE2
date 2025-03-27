import os
import pygame
import logging
from eve.config.display import Emotion

def generate_emotion_images(size=(800, 480)):
    """Generate basic emotion images using pygame shapes."""
    pygame.init()
    
    # Create assets directory if it doesn't exist
    assets_dir = "assets/emotions"
    os.makedirs(assets_dir, exist_ok=True)
    
    # Basic eye parameters
    eye_width = size[0] // 8
    eye_height = size[1] // 4
    eye_spacing = eye_width * 2
    center_y = size[1] // 2
    
    # Emotion-specific parameters
    emotion_params = {
        Emotion.NEUTRAL: {
            'eye_shape': 'circle',
            'eye_color': (255, 255, 255),
            'pupil_color': (0, 0, 0),
            'pupil_pos': (0, 0),  # Center
        },
        Emotion.HAPPY: {
            'eye_shape': 'circle',
            'eye_color': (255, 255, 255),
            'pupil_color': (0, 0, 0),
            'pupil_pos': (0, -eye_height//4),  # Looking up
        },
        Emotion.SAD: {
            'eye_shape': 'circle',
            'eye_color': (255, 255, 255),
            'pupil_color': (0, 0, 0),
            'pupil_pos': (0, eye_height//4),  # Looking down
        },
        Emotion.ANGRY: {
            'eye_shape': 'triangle',
            'eye_color': (255, 255, 255),
            'pupil_color': (255, 0, 0),
            'pupil_pos': (0, 0),
        },
        Emotion.SURPRISED: {
            'eye_shape': 'circle',
            'eye_color': (255, 255, 255),
            'pupil_color': (0, 0, 255),
            'pupil_pos': (0, 0),
            'scale': 1.2,
        },
        Emotion.CONFUSED: {
            'eye_shape': 'circle',
            'eye_color': (255, 255, 255),
            'pupil_color': (0, 0, 0),
            'pupil_pos': (eye_width//4, 0),  # Looking right
        },
    }
    
    for emotion in Emotion:
        # Create surface
        surface = pygame.Surface(size)
        surface.fill((0, 0, 0))  # Black background
        
        params = emotion_params[emotion]
        scale = params.get('scale', 1.0)
        
        # Calculate eye positions
        left_center = (size[0]//2 - eye_spacing//2, center_y)
        right_center = (size[0]//2 + eye_spacing//2, center_y)
        
        for center in [left_center, right_center]:
            # Draw eye background
            if params['eye_shape'] == 'circle':
                pygame.draw.ellipse(surface, params['eye_color'],
                                  (center[0] - eye_width//2 * scale,
                                   center[1] - eye_height//2 * scale,
                                   eye_width * scale,
                                   eye_height * scale))
            elif params['eye_shape'] == 'triangle':
                points = [
                    (center[0], center[1] - eye_height//2),
                    (center[0] - eye_width//2, center[1] + eye_height//2),
                    (center[0] + eye_width//2, center[1] + eye_height//2),
                ]
                pygame.draw.polygon(surface, params['eye_color'], points)
            
            # Draw pupil
            pupil_size = min(eye_width, eye_height) // 3
            pupil_pos = (
                center[0] + params['pupil_pos'][0],
                center[1] + params['pupil_pos'][1]
            )
            pygame.draw.circle(surface, params['pupil_color'],
                             pupil_pos, pupil_size)
        
        # Save the image
        filename = os.path.join(assets_dir, f"{emotion.name.lower()}.png")
        pygame.image.save(surface, filename)
        logging.info(f"Generated {filename}")
    
    pygame.quit()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_emotion_images()
    logging.info("All emotion assets generated successfully") 