import logging

logger = logging.getLogger(__name__)

class SpeechRecorder:
    """Records audio from microphone for speech processing"""
    
    def __init__(self):
        logger.info("Initializing speech recorder")
        self.is_recording = False
    
    def start_recording(self):
        """Start recording from microphone"""
        self.is_recording = True
        logger.info("Started speech recording")
        return True
        
    def stop_recording(self):
        """Stop recording and return the audio data"""
        self.is_recording = False
        logger.info("Stopped speech recording")
        # Return empty audio data for now
        return b''
    
    def is_active(self):
        """Check if recording is active"""
        return self.is_recording 