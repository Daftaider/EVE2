# EVE2

EVE2 is a Python-based system for object detection and display management.

## Features

- Camera input handling
- Object detection
- LCD display control
- Emotion display
- Real-time video processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EVE2.git
cd EVE2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main orchestrator:
```bash
python src/orchestrator.py
```

## Project Structure

```
EVE2/
├── src/
│   ├── vision/
│   │   ├── camera.py
│   │   └── object_detector.py
│   ├── display/
│   │   └── lcd_controller.py
│   └── orchestrator.py
├── requirements.txt
└── README.md
```

## Dependencies

- numpy>=1.21.0
- opencv-python>=4.5.0
- pygame>=2.0.0

## License

MIT License
