import logging
import threading
import time
import numpy as np
import queue
import random

logger = logging.getLogger(__name__)

class AudioCapture:
    """Records audio from microphone for speech processing"""
    
    def __init__(self, device_index=None, sample_rate=16000, channels=1, chunk_size=1024, 
                 threshold=0.01, mock_if_failed=True):
        """Initialize audio capture with fallback to mock"""
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.running = False
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=100)  # Limit queue size
        self.latest_audio = None  # Store latest audio data
        self.last_audio_time = 0  # Track when audio was last updated
        
        try:
            import pyaudio
            self.pa = pyaudio.PyAudio()
            
            # Validate device_index
            if device_index is not None:
                try:
                    device_index = int(device_index)
                except (TypeError, ValueError):
                    self.logger.warning(f"Invalid device_index: {device_index}, using default")
                    device_index = None
            
            # Try to open the audio stream
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.mock_mode = False
            
        except Exception as e:
            self.logger.error(f"Failed to open audio stream: {e}")
            if mock_if_failed:
                self.logger.info("Using mock audio capture")
                self.mock_mode = True
                self.mock_audio_thread = None
            else:
                raise

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

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for real audio capture"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Check if audio level is above threshold
            if np.abs(audio_data).mean() > self.threshold:
                if not self.audio_queue.full():
                    self.audio_queue.put_nowait(audio_data)
                    self.latest_audio = audio_data
                    self.last_audio_time = time.time()
                else:
                    self.logger.warning("Audio queue full, dropping data")
                    
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
            
        return (in_data, pyaudio.paContinue)

    def _generate_mock_audio(self):
        """Generate mock audio data"""
        while self.running:
            try:
                # Generate silent audio most of the time
                if random.random() < 0.1:  # 10% chance of "speech"
                    # Generate mock speech-like audio
                    duration = random.uniform(0.5, 2.0)  # Random duration between 0.5-2 seconds
                    t = np.linspace(0, duration, int(self.sample_rate * duration))
                    # Generate a mix of frequencies to simulate speech
                    frequencies = [random.uniform(80, 255) for _ in range(3)]
                    mock_audio = np.zeros_like(t)
                    for freq in frequencies:
                        mock_audio += np.sin(2 * np.pi * freq * t)
                    mock_audio *= 0.3  # Reduce amplitude
                    
                    # Split into chunks and add to queue
                    for i in range(0, len(mock_audio), self.chunk_size):
                        chunk = mock_audio[i:i + self.chunk_size]
                        if len(chunk) == self.chunk_size and not self.audio_queue.full():
                            chunk = chunk.astype(np.float32)
                            self.audio_queue.put_nowait(chunk)
                            self.latest_audio = chunk
                            self.last_audio_time = time.time()
                
                time.sleep(0.1)  # Prevent tight loop
                
            except Exception as e:
                self.logger.error(f"Error generating mock audio: {e}")
                time.sleep(1)

    def start(self):
        """Start audio capture"""
        self.running = True
        self.last_audio_time = time.time()  # Reset timer
        if self.mock_mode:
            self.mock_audio_thread = threading.Thread(target=self._generate_mock_audio)
            self.mock_audio_thread.daemon = True
            self.mock_audio_thread.start()
        elif self.stream:
            self.stream.start_stream()

    def stop(self):
        """Stop audio capture"""
        self.running = False
        if self.mock_mode and self.mock_audio_thread:
            self.mock_audio_thread.join(timeout=1.0)
        elif self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()
        self.clear_audio()

    def is_active(self):
        """Check if audio capture is active"""
        return self.running and (
            (self.mock_mode and self.mock_audio_thread and self.mock_audio_thread.is_alive()) or
            (not self.mock_mode and self.stream and self.stream.is_active())
        )

    def set_threshold(self, threshold):
        """Set the audio detection threshold"""
        self.threshold = max(0.0, float(threshold))
        logger.info(f"Audio detection threshold set to: {self.threshold}")

# For backward compatibility
SpeechRecorder = AudioCapture 