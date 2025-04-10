import pygame
import time
import os

def main():
    # Initialize pygame
    pygame.init()
    
    # Set up display
    width, height = 800, 480
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        try:
            # Load and display the current frame
            if os.path.exists("current_display.png"):
                image = pygame.image.load("current_display.png")
                screen.blit(image, (0, 0))
                pygame.display.flip()
        except Exception as e:
            print(f"Error updating display: {e}")
        
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main() 