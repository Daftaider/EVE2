#!/usr/bin/env python3
import os
import sys
import time
import pygame

def test_display_modes():
    """Test various display initialization methods"""
    print("EVE2 Display Diagnostic Tool")
    print("-" * 40)
    
    # Test environment
    print("\nSystem Information:")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Pygame version: {pygame.version.ver}")
    print(f"Display env var: {'DISPLAY' in os.environ}")
    if 'DISPLAY' in os.environ:
        print(f"Display value: {os.environ['DISPLAY']}")
    
    # Try standard mode
    print("\nTesting standard display mode...")
    try:
        pygame.init()
        pygame.display.init()
        screen = pygame.display.set_mode((800, 480))
        print("✓ Standard mode works!")
        
        # Test display update
        screen.fill((0, 0, 255))
        pygame.display.flip()
        print("✓ Display updates work!")
        pygame.quit()
    except Exception as e:
        print(f"✗ Standard mode failed: {e}")
    
    # Try software mode
    print("\nTesting software rendering mode...")
    try:
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        os.environ['SDL_RENDERER_DRIVER'] = 'software'
        
        pygame.init()
        pygame.display.init()
        screen = pygame.display.set_mode((800, 480), pygame.SWSURFACE)
        print("✓ Software mode works!")
        
        # Test display update
        screen.fill((0, 255, 0))
        pygame.display.flip()
        print("✓ Display updates work!")
        pygame.quit()
    except Exception as e:
        print(f"✗ Software mode failed: {e}")
    
    # Try dummy mode
    print("\nTesting dummy display mode...")
    try:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        pygame.init()
        pygame.display.init()
        screen = pygame.Surface((800, 480))
        print("✓ Dummy mode works!")
        pygame.quit()
    except Exception as e:
        print(f"✗ Dummy mode failed: {e}")
    
    print("\nDisplay Diagnostics Complete")
    print("If standard and software modes failed but dummy mode works,")
    print("the system can run EVE2 in fallback mode with no visual output.")

if __name__ == "__main__":
    test_display_modes() 