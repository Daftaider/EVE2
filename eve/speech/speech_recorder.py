import logging
import threading
import time
import numpy as np

logger = logging.getLogger(__name__)

class AudioCapture:
    """Records audio from microphone for speech processing"""
    
    def __init__(self, device_index=None, sample_rate=16000, channels=1):
        """Initialize the audio capture system"""
        logger.info("Initializing audio capture")
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_buffer = []
        self._recording_thread = None
    
    def start_recording(self):
        """Start recording from microphone"""
        if self.is_recording:
            logger.warning("Recording is already active")
            return False
            
        self.is_recording = True
        self.audio_buffer = []
        self._recording_thread = threading.Thread(target=self._record_audio)
        self._recording_thread.daemon = True
        self._recording_thread.start()
        logger.info("Started audio recording")
        return True
        
    def stop_recording(self):
        """Stop recording and return the audio data"""
        if not self.is_recording:
            logger.warning("No active recording to stop")
            return np.array([])
            
        self.is_recording = False
        if self._recording_thread:
            self._recording_thread.join(timeout=1.0)
            
        # Convert buffer to numpy array (simulated here)
        audio_data = np.concatenate(self.audio_buffer) if self.audio_buffer else np.array([])
        logger.info(f"Stopped recording, captured {len(audio_data)} samples")
        return audio_data
    
    def is_active(self):
        """Check if recording is active"""
        return self.is_recording
        
    def _record_audio(self):
        """Recording thread function (simulation)"""
        try:
            while self.is_recording:
                # Simulate capturing audio - in real implementation, this would use PyAudio or similar
                audio_chunk = np.zeros(self.sample_rate // 10, dtype=np.float32)  # 100ms of silence
                self.audio_buffer.append(audio_chunk)
                time.sleep(0.1)  # Simulate real-time recording
        except Exception as e:
            logger.error(f"Error in recording thread: {e}")
            self.is_recording = False

# For backward compatibility
SpeechRecorder = AudioCapture 