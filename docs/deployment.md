# EVE2 Deployment Guide

This document covers how to deploy the EVE2 system in both standalone and distributed configurations.

## Prerequisites

- Raspberry Pi 5 (or multiple Pis for distributed setup)
- Camera module compatible with Raspberry Pi
- Microphone (USB or Pi-compatible)
- LCD Screen for eye animations
- Speakers for audio output
- Python 3.9+ installed
- Required Python packages (see `requirements.txt`)

## Standalone Deployment

In standalone mode, a single Raspberry Pi runs all components of the EVE2 system:

1. Clone the repository
   ```
   git clone https://github.com/yourusername/eve2.git
   cd eve2
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Download the required models
   ```
   mkdir -p models/vosk models/llm
   # Download Vosk model (small English model)
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   unzip vosk-model-small-en-us-0.15.zip -d models/vosk/
   
   # Download LLaMA model (quantized for Pi)
   wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O models/llm/llama-2-7b-chat.Q4_K_M.gguf
   ```

4. Configure the system
   Edit `eve/config.py` to match your hardware setup.

5. Run the system
   ```
   python scripts/start_eve.py
   ```

## Distributed Deployment

For better performance, EVE2 can be distributed across multiple Raspberry Pis, with each Pi handling specific roles:

### Role Configuration

- **Vision Pi**: Handles camera input, face detection, and emotion analysis
- **Speech Pi**: Handles audio processing, speech recognition, LLM, and TTS
- **Display Pi**: Controls the LCD screen and animations
- **Master Pi**: Coordinates communication between other Pis (can be combined with one of the above)

### Network Setup

1. Ensure all Pis are on the same network and can communicate with each other
2. Determine the IP address of each Pi using `hostname -I`
3. Choose one Pi to be the Master (usually the one with the most resources)

### Deployment Steps

1. Clone the repository on all Pis
   ```
   git clone https://github.com/yourusername/eve2.git
   cd eve2
   ```

2. Install dependencies on all Pis
   ```
   pip install -r requirements.txt
   ```

3. Download the required models on the relevant Pis
   - On the Speech Pi: Download Vosk and LLaMA models
   - On the Vision Pi: No model downloads required (using face_recognition and fer libraries)

4. Configure and start the Master Pi
   ```
   python scripts/start_eve.py --role master --distributed
   ```

5. Configure and start the Vision Pi (replace MASTER_IP with the Master Pi's IP address)
   ```
   python scripts/start_eve.py --role vision --distributed --master-ip MASTER_IP
   ```

6. Configure and start the Speech Pi
   ```
   python scripts/start_eve.py --role speech --distributed --master-ip MASTER_IP
   ```

7. Configure and start the Display Pi
   ```
   python scripts/start_eve.py --role display --distributed --master-ip MASTER_IP
   ```

## Command-Line Options

EVE2 provides various command-line options for configuration:

### General Options
- `--role {all,vision,speech,display,master}`: Set the role of this instance
- `--distributed`: Enable distributed mode
- `--master-ip IP`: IP address of the master node (for distributed mode)
- `--master-port PORT`: Port of the master node (for distributed mode)

### Hardware Options
- `--camera-index INDEX`: Camera device index
- `--no-camera`: Disable camera
- `--no-display`: Disable display
- `--fullscreen`: Run display in fullscreen mode
- `--no-audio-input`: Disable audio input
- `--no-audio-output`: Disable audio output
- `--audio-input-device INDEX`: Audio input device index
- `--audio-output-device INDEX`: Audio output device index

### Logging Options
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set the logging level
- `--log-file PATH`: Path to log file

## Testing the Installation

After deployment, you can test the system with these steps:

1. Check if the system is running:
   ```
   ps aux | grep start_eve.py
   ```

2. Verify logs for any errors:
   ```
   cat logs/eve.log
   ```

3. Test basic functionality:
   - Position yourself in front of the camera and check if your face is detected
   - Check if the LCD display shows eye animations
   - Speak to the system and verify it responds

## Troubleshooting

- **Camera not working**: Check if the camera module is properly connected and enabled in Raspberry Pi configuration
- **Audio issues**: Verify audio devices using `arecord -l` and `aplay -l`, and update `config.py` or command-line options accordingly
- **Communication errors in distributed mode**: Check network connectivity between Pis using `ping`
- **High CPU usage**: Consider distributing the system across multiple Pis or reducing the resolution of camera input and display

## System Monitoring

To monitor the system's performance:
```
# Check CPU and memory usage
htop

# Monitor temperature
vcgencmd measure_temp

# Check available disk space
df -h
```

## Service Setup (Optional)

To run EVE2 as a service that starts automatically on boot:

1. Create a systemd service file:
   ```
   sudo nano /etc/systemd/system/eve2.service
   ```

2. Add the following content:
   ```
   [Unit]
   Description=EVE2 Robot System
   After=network.target

   [Service]
   User=pi
   WorkingDirectory=/home/pi/eve2
   ExecStart=/usr/bin/python3 /home/pi/eve2/scripts/start_eve.py
   Restart=on-failure
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:
   ```
   sudo systemctl enable eve2.service
   sudo systemctl start eve2.service
   ```

4. Check service status:
   ```
   sudo systemctl status eve2.service
   ``` 