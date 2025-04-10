"""
EVE2 - Interactive Robotic System
"""

__version__ = "0.1.0"
__author__ = "EVE2 Team"

import os
import sys

# Add package root directory to Python path
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_root not in sys.path:
    sys.path.insert(0, package_root) 