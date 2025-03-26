import logging
import threading
import time
import numpy as np
import queue

logger = logging.getLogger(__name__)

class AudioCapture:
    """Records audio from microphone for speech processing"""
    
    def __init__(self, device_index=None, sample_rate=16000, channels=1, chunk_size=1024):
        """Initialize audio capture system"""
        logger.info("Initializing audio capture")
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self._recording_thread = None
        self._stream = None
        self._last_check_time = time.time()
        self._audio_threshold = 0.01  # Threshold for detecting non-silent audio
        
    def has_new_audio(self):
        """Check if new audio data is available and above noise threshold"""
        try:
            # Peek at the queue without removing the data
            audio_data = self.audio_queue.queue[0] if not self.audio_queue.empty() else None
            
            if audio_data is not None:
                # Check if audio is above threshold (not just silence)
                audio_level = np.abs(audio_data).mean()
                return audio_level > self._audio_threshold
            return False
        except Exception as e:
            logger.error(f"Error checking for new audio: {e}")
            return False
    
    def get_audio_level(self):
        """Get current audio level (for monitoring)"""
        try:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.queue[0]
                return np.abs(audio_data).mean()
            return 0.0
        except Exception as e:
            logger.error(f"Error getting audio level: {e}")
            return 0.0
    
    def set_threshold(self, threshold):
        """Set the audio detection threshold"""
        self._audio_threshold = max(0.0, float(threshold))
        logger.info(f"Audio detection threshold set to: {self._audio_threshold}")

    def start(self):
        """Start audio capture in a separate thread"""
        if self.is_recording:
            logger.warning("Audio capture already running")
            return False
            
        self.is_recording = True
        self._recording_thread = threading.Thread(target=self._capture_loop)
        self._recording_thread.daemon = True
        self._recording_thread.start()
        logger.info("Started audio capture")
        return True
        
    def stop(self):
        """Stop audio capture"""
        if not self.is_recording:
            logger.warning("Audio capture not running")
            return False
            
        self.is_recording = False
        if self._recording_thread:
            self._recording_thread.join(timeout=1.0)
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
        self._stream = None
        logger.info("Stopped audio capture")
        return True
        
    def get_audio(self, timeout=0.1):
        """Get captured audio data from the queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _capture_loop(self):
        """Main audio capture loop"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            # Try to open the audio stream
            try:
                self._stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk_size
                )
                logger.info("Opened audio stream successfully")
            except Exception as e:
                logger.error(f"Failed to open audio stream: {e}")
                self._use_mock_audio()
                return
                
            # Main capture loop
            while self.is_recording:
                try:
                    data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    
                    # Keep queue from growing too large
                    while self.audio_queue.qsize() > 10:
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                            
                    self.audio_queue.put(audio_data)
                except Exception as e:
                    logger.error(f"Error capturing audio: {e}")
                    time.sleep(0.1)
                    
        except ImportError:
            logger.warning("PyAudio not available, using mock audio capture")
            self._use_mock_audio()
        except Exception as e:
            logger.error(f"Error in audio capture loop: {e}")
            self._use_mock_audio()
        finally:
            if self._stream:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except:
                    pass
            self._stream = None
            
    def _use_mock_audio(self):
        """Generate mock audio data when real capture fails"""
        logger.info("Using mock audio capture")
        while self.is_recording:
            # Generate mock audio data with occasional "speech-like" patterns
            t = time.time()
            if int(t) % 10 < 3:  # Simulate speech every 10 seconds for 3 seconds
                # Generate "speech-like" audio
                mock_data = np.random.normal(0, 0.1, self.chunk_size).astype(np.float32)
            else:
                # Generate silence with very low noise
                mock_data = np.random.normal(0, 0.001, self.chunk_size).astype(np.float32)
                
            self.audio_queue.put(mock_data)
            time.sleep(self.chunk_size / self.sample_rate)  # Simulate real-time capture
            
    def is_active(self):
        """Check if audio capture is active"""
        return self.is_recording

# For backward compatibility
SpeechRecorder = AudioCapture 