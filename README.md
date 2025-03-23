
# EVE2 - Emotional Virtual Entity

EVE2 is an interactive robotic system inspired by EVE from Wall-E, designed to run on Raspberry Pi hardware. It features face detection, emotion recognition, interactive LCD-based eye animations, and natural language conversation capabilities.

## Features

- **Face Detection and Recognition**: Identifies faces and recognizes known individuals
- **Emotion Analysis**: Detects facial expressions and analyzes emotions
- **Expressive Eyes**: Renders dynamic eye animations on an LCD display based on detected emotions
- **Speech Recognition**: Converts spoken language to text
- **Language Model Interaction**: Processes text using a local LLM to generate natural responses
- **Text-to-Speech**: Converts responses to natural-sounding speech
- **Distributed Architecture**: Can run as a single system or distributed across multiple Raspberry Pis

## System Requirements

### Hardware

- Raspberry Pi 5 (recommended) or Raspberry Pi 4 with at least 4GB RAM
- Camera module (Raspberry Pi Camera v2 or better)
- LCD display (800x480 recommended)
- Microphone (USB or I2S)
- Speakers or headphones
- Power supply (adequate for the Pi and peripherals)

### Software

- Raspberry Pi OS (64-bit recommended)
- Python 3.9+
- Required libraries (see `requirements.txt`)

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/eve2.git
   cd eve2
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models**:
   ```bash
   python scripts/download_models.py
   ```

5. **Configure the system**:
   Edit `config.yaml` to match your hardware and preferences.

## Usage

### Running EVE2

To start the system with default settings:

```bash
python scripts/start_eve.py
```

For additional options:

```bash
python scripts/start_eve.py --help
```

### Common Command-Line Options

- `--log-level DEBUG|INFO|WARNING|ERROR`: Set logging level
- `--no-camera`: Disable camera/vision module
- `--no-display`: Disable LCD display
- `--no-audio-input`: Disable microphone/speech recognition
- `--no-audio-output`: Disable speakers/text-to-speech
- `--fullscreen`: Run in fullscreen mode
- `--camera-index N`: Specify camera device index
- `--distributed`: Enable distributed mode
- `--role standalone|master|vision|speech|display`: Set role in distributed mode

## Architecture

EVE2 is built with a modular architecture consisting of four main components:

1. **Vision Module**: Handles camera input, face detection, and emotion analysis
2. **Display Module**: Controls the LCD screen and renders eye animations
3. **Speech Module**: Processes audio input, speech recognition, LLM response generation, and text-to-speech
4. **Communication Module**: Enables message passing between modules and supports distributed deployment

These components can run on a single Raspberry Pi or be distributed across multiple devices for improved performance.

See the [architecture documentation](docs/architecture.md) for more details.

## Configuration

EVE2 is highly configurable through the `config.yaml` file, which allows you to adjust:

- Hardware settings (camera, display, audio)
- Face detection and emotion analysis parameters
- Speech recognition and TTS settings
- LLM configuration
- Logging options
- Network settings for distributed mode

## Extending EVE2

### Adding Known Faces

Place photos of people to recognize in the `data/known_faces` directory, with filenames matching their names (e.g., `john.jpg`).

### Customizing Emotions

Edit the emotion mappings in the config file to customize how emotions are detected and displayed.

### Changing LLM Model

You can replace the default LLM model with any GGUF-format model compatible with llama.cpp. Adjust the `llm_model` setting in the config file.

## Deployment

For detailed deployment instructions, including standalone and distributed setups, see the [deployment documentation](docs/deployment.md).

## Contributors

- Daftaider <richard@lazyadmin.uk>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by EVE from Wall-E
- Built with open-source AI models and tools 
=======
# EVE2
>>>>>>> a443b25dd076b3d7f9e66517b9faffda4f9bdb3d
