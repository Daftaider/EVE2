import logging
import threading
import time
import numpy as np
import queue
import random

logger = logging.getLogger(__name__)

class AudioCapture:
    """Records audio from microphone for speech processing"""
    
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024, threshold=0.01, **kwargs):
        """Initialize audio capture with mock implementation
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            chunk_size: Size of audio chunks
            threshold: Audio detection threshold
            **kwargs: Additional arguments (ignored)
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.running = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.latest_audio = None
        self.last_audio_time = 0
        self.mock_audio_thread = None
        self.logger.info("Using mock audio capture")

    def get_latest_audio(self):
        """Get the most recent audio data"""
        current_time = time.time()
        
        # Check if we have new audio data in the queue
        while not self.audio_queue.empty():
            try:
                self.latest_audio = self.audio_queue.get_nowait()
                self.last_audio_time = current_time
            except queue.Empty:
                break
        
        # Return None if audio is too old (more than 1 second)
        if current_time - self.last_audio_time > 1.0:
            return None
            
        return self.latest_audio

    def get_audio(self):
        """Get next audio chunk from queue (maintains backwards compatibility)"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def has_new_audio(self):
        """Check if new audio data is available"""
        if self.audio_queue.empty():
            return False
            
        # Also check if latest audio is recent (within last second)
        current_time = time.time()
        return (current_time - self.last_audio_time) <= 1.0

    def get_audio_level(self):
        """Get current audio level"""
        audio_data = self.get_latest_audio()
        if audio_data is not None:
            return np.abs(audio_data).mean()
        return 0.0

    def clear_audio(self):
        """Clear any buffered audio data"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.latest_audio = None
        self.last_audio_time = 0

    def _generate_mock_audio(self):
        """Generate mock audio data"""
        while self.running:
            try:
                # Generate silent audio most of the time
                if random.random() < 0.1:  # 10% chance of "speech"
                    # Generate mock speech-like audio
                    duration = random.uniform(0.5, 2.0)  # Random duration between 0.5-2 seconds
                    samples = int(self.sample_rate * duration)
                    t = np.linspace(0, duration, samples)
                    # Generate a mix of frequencies to simulate speech
                    mock_audio = np.zeros_like(t)
                    for _ in range(3):
                        freq = random.uniform(80, 255)
                        mock_audio += np.sin(2 * np.pi * freq * t)
                    mock_audio *= 0.3  # Reduce amplitude
                    
                    # Split into chunks and add to queue
                    for i in range(0, len(mock_audio), self.chunk_size):
                        if not self.running:
                            break
                        chunk = mock_audio[i:i + self.chunk_size]
                        if len(chunk) == self.chunk_size and not self.audio_queue.full():
                            chunk = chunk.astype(np.float32)
                            self.audio_queue.put_nowait(chunk)
                            self.latest_audio = chunk
                            self.last_audio_time = time.time()
                
                time.sleep(0.1)  # Prevent tight loop
                
            except Exception as e:
                self.logger.error("Error generating mock audio: {}".format(str(e)))
                time.sleep(1)

    def start(self):
        """Start audio capture"""
        self.running = True
        self.last_audio_time = time.time()  # Reset timer
        self.mock_audio_thread = threading.Thread(target=self._generate_mock_audio)
        self.mock_audio_thread.daemon = True
        self.mock_audio_thread.start()
        self.logger.info("Audio capture started")

    def stop(self):
        """Stop audio capture"""
        self.running = False
        if self.mock_audio_thread:
            self.mock_audio_thread.join(timeout=1.0)
        self.clear_audio()
        self.logger.info("Audio capture stopped")

    def is_active(self):
        """Check if audio capture is active"""
        return self.running and self.mock_audio_thread and self.mock_audio_thread.is_alive()

    def set_threshold(self, threshold):
        """Set the audio detection threshold"""
        self.threshold = max(0.0, float(threshold))
        logger.info(f"Audio detection threshold set to: {self.threshold}")

# For backward compatibility
SpeechRecorder = AudioCapture 